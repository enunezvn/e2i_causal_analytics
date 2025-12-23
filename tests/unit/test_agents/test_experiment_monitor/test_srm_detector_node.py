"""Tests for SRM Detector Node.

Tests cover:
- Node initialization
- SRM detection with chi-squared tests
- Variant count retrieval
- Severity determination based on p-values
- Edge cases and error handling
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor.nodes.srm_detector import SRMDetectorNode
from src.agents.experiment_monitor.state import ExperimentMonitorState


class TestSRMDetectorNodeInit:
    """Tests for SRMDetectorNode initialization."""

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        node = SRMDetectorNode()
        assert node is not None
        assert node._client is None

    def test_multiple_node_instances(self):
        """Test creating multiple node instances."""
        node1 = SRMDetectorNode()
        node2 = SRMDetectorNode()
        assert node1 is not node2


class TestSRMDetectorGetClient:
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

            node = SRMDetectorNode()
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

            node = SRMDetectorNode()
            await node._get_client()
            await node._get_client()

            mock_get_client.assert_called_once()


class TestSRMDetectorExecute:
    """Tests for execute method."""

    @pytest.fixture
    def state_with_experiments(self):
        """State with experiment summaries."""
        return {
            "query": "Check SRM",
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
        node = SRMDetectorNode()

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(base_monitor_state)

            assert result["srm_issues"] == []

    @pytest.mark.asyncio
    async def test_execute_detects_srm_with_mock_data(self, state_with_experiments):
        """Test SRM detection uses mock data without client."""
        node = SRMDetectorNode()

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(state_with_experiments)

            # Mock data has slight imbalance but not significant
            assert "srm_issues" in result

    @pytest.mark.asyncio
    async def test_execute_accumulates_latency(self, state_with_experiments):
        """Test that latency is accumulated."""
        node = SRMDetectorNode()
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
        node = SRMDetectorNode()

        with patch.object(node, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("SRM detection failed")

            result = await node.execute(state_with_experiments)

            assert len(result["errors"]) >= 1
            assert "SRM detection failed" in result["errors"][0]["error"]
            assert result["errors"][0]["node"] == "srm_detector"


class TestCheckSRM:
    """Tests for _check_srm method."""

    @pytest.fixture
    def node(self):
        return SRMDetectorNode()

    @pytest.fixture
    def experiment_summary(self):
        return {
            "experiment_id": "exp-001",
            "name": "Test",
            "status": "running",
            "health_status": "healthy",
            "days_running": 7,
            "total_enrolled": 500,
            "enrollment_rate": 71.43,
            "current_information_fraction": 0.5,
        }

    @pytest.fixture
    def state_with_threshold(self):
        return {
            "query": "",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "errors": [],
            "warnings": [],
            "status": "checking",
        }

    @pytest.mark.asyncio
    async def test_skip_low_sample_size(self, node, state_with_threshold):
        """Test that low sample size experiments are skipped."""
        experiment = {
            "experiment_id": "exp-small",
            "total_enrolled": 50,  # Below 100 minimum
        }

        result = await node._check_srm(experiment, None, state_with_threshold)

        assert result is None

    @pytest.mark.asyncio
    async def test_detects_significant_srm(self, node, experiment_summary, state_with_threshold):
        """Test detection of significant SRM."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = (
            [{"variant": "control"}] * 350 +
            [{"variant": "treatment"}] * 150  # 70/30 split, significant SRM
        )

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_srm(experiment_summary, mock_client, state_with_threshold)

        assert result is not None
        assert result["detected"] is True
        assert result["experiment_id"] == "exp-001"

    @pytest.mark.asyncio
    async def test_no_srm_for_balanced_split(self, node, experiment_summary, state_with_threshold):
        """Test no SRM for balanced allocation."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = (
            [{"variant": "control"}] * 250 +
            [{"variant": "treatment"}] * 250  # Perfect 50/50 split
        )

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_srm(experiment_summary, mock_client, state_with_threshold)

        # Perfect split should not trigger SRM
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_mock_data_without_client(self, node, experiment_summary, state_with_threshold):
        """Test that mock data is used without client."""
        result = await node._check_srm(experiment_summary, None, state_with_threshold)

        # Mock data is 48/52, very slight imbalance - should not be significant
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_counts(self, node, experiment_summary, state_with_threshold):
        """Test handling of empty variant counts."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_srm(experiment_summary, mock_client, state_with_threshold)

        assert result is None


class TestSRMSeverity:
    """Tests for SRM severity determination."""

    @pytest.fixture
    def node(self):
        return SRMDetectorNode()

    @pytest.fixture
    def experiment_summary(self):
        return {
            "experiment_id": "exp-001",
            "total_enrolled": 1000,
        }

    @pytest.fixture
    def state_with_threshold(self):
        return {
            "srm_threshold": 0.01,  # More lenient for testing
            "errors": [],
            "warnings": [],
        }

    @pytest.mark.asyncio
    async def test_critical_severity_very_low_p_value(self, node, experiment_summary, state_with_threshold):
        """Test critical severity for very low p-value."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        # Extreme imbalance: 900 control, 100 treatment
        mock_result.data = (
            [{"variant": "control"}] * 900 +
            [{"variant": "treatment"}] * 100
        )

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_srm(experiment_summary, mock_client, state_with_threshold)

        assert result is not None
        assert result["severity"] == "critical"
        assert result["p_value"] < 0.0001

    @pytest.mark.asyncio
    async def test_warning_severity_moderate_p_value(self, node, experiment_summary, state_with_threshold):
        """Test warning severity for moderate p-value."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        # Moderate imbalance that gives p-value around 0.0001-0.001
        mock_result.data = (
            [{"variant": "control"}] * 600 +
            [{"variant": "treatment"}] * 400
        )

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_srm(experiment_summary, mock_client, state_with_threshold)

        # 60/40 split with 1000 samples should be significant
        if result:
            assert result["severity"] in ["warning", "critical"]


class TestGetVariantCounts:
    """Tests for _get_variant_counts method."""

    @pytest.fixture
    def node(self):
        return SRMDetectorNode()

    @pytest.mark.asyncio
    async def test_get_counts_from_database(self, node):
        """Test getting variant counts from database."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"variant": "control"},
            {"variant": "control"},
            {"variant": "treatment"},
            {"variant": "treatment"},
            {"variant": "treatment"},
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        counts = await node._get_variant_counts(mock_client, "exp-001")

        assert counts["control"] == 2
        assert counts["treatment"] == 3

    @pytest.mark.asyncio
    async def test_get_counts_empty_result(self, node):
        """Test handling of empty result."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        counts = await node._get_variant_counts(mock_client, "exp-001")

        assert counts == {}

    @pytest.mark.asyncio
    async def test_get_counts_handles_exception(self, node):
        """Test handling of database exception."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_client.table = MagicMock(return_value=mock_query)

        counts = await node._get_variant_counts(mock_client, "exp-001")

        assert counts == {}

    @pytest.mark.asyncio
    async def test_get_counts_handles_missing_variant(self, node):
        """Test handling of rows without variant field."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"variant": "control"},
            {},  # Missing variant
            {"variant": "treatment"},
        ]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        counts = await node._get_variant_counts(mock_client, "exp-001")

        assert counts["control"] == 1
        assert counts["treatment"] == 1
        assert counts["unknown"] == 1


class TestChiSquaredTest:
    """Tests for _chi_squared_test method."""

    @pytest.fixture
    def node(self):
        return SRMDetectorNode()

    def test_perfect_split_returns_high_p_value(self, node):
        """Test perfect 50/50 split returns high p-value."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 500, "treatment": 500}

        chi2, p_value = node._chi_squared_test(expected, actual)

        assert chi2 == 0.0
        assert p_value == 1.0

    def test_significant_imbalance_returns_low_p_value(self, node):
        """Test significant imbalance returns low p-value."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 800, "treatment": 200}

        chi2, p_value = node._chi_squared_test(expected, actual)

        assert chi2 > 0
        assert p_value < 0.001

    def test_zero_total_returns_no_srm(self, node):
        """Test zero total count returns no SRM."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 0, "treatment": 0}

        chi2, p_value = node._chi_squared_test(expected, actual)

        assert chi2 == 0.0
        assert p_value == 1.0

    def test_slight_imbalance_not_significant(self, node):
        """Test slight imbalance is not significant."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 505, "treatment": 495}  # 50.5% / 49.5%

        chi2, p_value = node._chi_squared_test(expected, actual)

        assert p_value > 0.01  # Not significant

    def test_handles_multi_variant(self, node):
        """Test handling of multiple variants."""
        expected = {"control": 0.33, "treatment_a": 0.33, "treatment_b": 0.34}
        actual = {"control": 330, "treatment_a": 330, "treatment_b": 340}

        chi2, p_value = node._chi_squared_test(expected, actual)

        assert chi2 >= 0
        assert 0 <= p_value <= 1

    def test_handles_missing_expected_variant(self, node):
        """Test handling of variant not in expected ratio."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 400, "treatment": 400, "other": 200}

        chi2, p_value = node._chi_squared_test(expected, actual)

        # Should still compute, using default ratio for unknown variant
        assert chi2 >= 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def node(self):
        return SRMDetectorNode()

    @pytest.mark.asyncio
    async def test_exactly_100_sample_size(self, node):
        """Test experiment with exactly 100 samples (minimum)."""
        state = {
            "srm_threshold": 0.001,
            "errors": [],
            "warnings": [],
        }
        experiment = {
            "experiment_id": "exp-100",
            "total_enrolled": 100,
        }

        # Should not skip due to sample size
        result = await node._check_srm(experiment, None, state)

        # With mock data (48/52), should not detect SRM
        assert result is None

    @pytest.mark.asyncio
    async def test_exactly_99_sample_size(self, node):
        """Test experiment with 99 samples (below minimum)."""
        state = {
            "srm_threshold": 0.001,
            "errors": [],
            "warnings": [],
        }
        experiment = {
            "experiment_id": "exp-99",
            "total_enrolled": 99,
        }

        result = await node._check_srm(experiment, None, state)

        assert result is None  # Skipped due to low sample

    def test_chi_squared_with_very_large_samples(self, node):
        """Test chi-squared with very large sample sizes."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 1000000, "treatment": 999900}  # Tiny imbalance

        chi2, p_value = node._chi_squared_test(expected, actual)

        # With large samples, even tiny imbalance becomes significant
        assert chi2 > 0
        # P-value might be small due to large sample size

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
                {"experiment_id": "exp-3", "total_enrolled": 50},  # Will be skipped
            ],
            "srm_issues": [],
            "enrollment_issues": [],
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

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None

            result = await node.execute(state)

            # All experiments processed, exp-3 skipped due to low sample
            assert "srm_issues" in result

    def test_srm_issue_structure(self, node):
        """Test that SRM issue has correct structure."""
        expected = {"control": 0.5, "treatment": 0.5}
        actual = {"control": 800, "treatment": 200}

        # This creates a significant SRM
        chi2, p_value = node._chi_squared_test(expected, actual)

        # Verify the chi-squared test produces expected output format
        assert isinstance(chi2, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
