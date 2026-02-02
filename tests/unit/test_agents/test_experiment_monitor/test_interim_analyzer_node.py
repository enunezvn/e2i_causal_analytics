"""Tests for Interim Analyzer Node.

Tests cover:
- Node initialization
- Interim analysis trigger detection
- Milestone checking logic
- Already-analyzed milestone detection
- Edge cases and error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor.nodes.interim_analyzer import InterimAnalyzerNode


class TestInterimAnalyzerNodeInit:
    """Tests for InterimAnalyzerNode initialization."""

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        node = InterimAnalyzerNode()
        assert node is not None
        assert node._client is None

    def test_default_milestones(self):
        """Test default milestone values."""
        node = InterimAnalyzerNode()
        assert node.DEFAULT_MILESTONES == [0.25, 0.50, 0.75]

    def test_multiple_node_instances(self):
        """Test creating multiple node instances."""
        node1 = InterimAnalyzerNode()
        node2 = InterimAnalyzerNode()
        assert node1 is not node2


class TestInterimAnalyzerGetClient:
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

            node = InterimAnalyzerNode()
            client = await node._get_client()

            assert client is mock_client
            mock_get_client.assert_called_once()


class TestInterimAnalyzerExecute:
    """Tests for execute method."""

    @pytest.fixture
    def state_with_experiments(self):
        """State with experiment summaries."""
        return {
            "query": "Check interim",
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
    async def test_execute_skips_when_check_interim_false(self, state_with_experiments):
        """Test that execution skips when check_interim is False."""
        state_with_experiments["check_interim"] = False
        node = InterimAnalyzerNode()

        result = await node.execute(state_with_experiments)

        assert result["interim_triggers"] == []

    @pytest.mark.asyncio
    async def test_execute_with_no_experiments(self, base_monitor_state):
        """Test execute with empty experiments list."""
        node = InterimAnalyzerNode()

        result = await node.execute(base_monitor_state)

        assert result["interim_triggers"] == []

    @pytest.mark.asyncio
    async def test_execute_detects_milestone(self, state_with_experiments):
        """Test that milestone is detected."""
        node = InterimAnalyzerNode()

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None  # Use mock path

            result = await node.execute(state_with_experiments)

            # 0.5 information fraction should trigger 50% milestone
            assert len(result["interim_triggers"]) == 1
            assert result["interim_triggers"][0]["milestone_reached"] == "50%"

    @pytest.mark.asyncio
    async def test_execute_accumulates_latency(self, state_with_experiments):
        """Test that latency is accumulated."""
        node = InterimAnalyzerNode()
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
        node = InterimAnalyzerNode()

        with patch.object(node, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Interim check failed")

            result = await node.execute(state_with_experiments)

            assert len(result["errors"]) >= 1
            assert "Interim check failed" in result["errors"][0]["error"]
            assert result["errors"][0]["node"] == "interim_analyzer"


class TestCheckInterimTrigger:
    """Tests for _check_interim_trigger method."""

    @pytest.fixture
    def node(self):
        return InterimAnalyzerNode()

    @pytest.mark.asyncio
    async def test_no_trigger_below_25_percent(self, node):
        """Test no trigger when below 25% milestone."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.20,  # Below 25%
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_trigger_at_25_percent(self, node):
        """Test trigger at 25% milestone."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.25,
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is not None
        assert result["milestone_reached"] == "25%"
        assert result["analysis_number"] == 1
        assert result["triggered"] is True

    @pytest.mark.asyncio
    async def test_trigger_at_50_percent(self, node):
        """Test trigger at 50% milestone."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.50,
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is not None
        assert result["milestone_reached"] == "50%"
        assert result["analysis_number"] == 2

    @pytest.mark.asyncio
    async def test_trigger_at_75_percent(self, node):
        """Test trigger at 75% milestone."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.75,
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is not None
        assert result["milestone_reached"] == "75%"
        assert result["analysis_number"] == 3

    @pytest.mark.asyncio
    async def test_highest_milestone_used(self, node):
        """Test that highest reached milestone is used."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.60,  # Between 50% and 75%
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is not None
        assert result["milestone_reached"] == "50%"  # Highest fully reached

    @pytest.mark.asyncio
    async def test_no_trigger_when_already_analyzed(self, node):
        """Test no trigger when milestone already analyzed."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"information_fraction": 0.50}]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.50,
        }

        result = await node._check_interim_trigger(experiment, mock_client)

        assert result is None


class TestMilestoneDetection:
    """Tests for milestone detection logic."""

    @pytest.fixture
    def node(self):
        return InterimAnalyzerNode()

    @pytest.mark.asyncio
    async def test_exactly_at_milestone_triggers(self, node):
        """Test trigger when exactly at milestone."""
        for milestone in [0.25, 0.50, 0.75]:
            experiment = {
                "experiment_id": "exp-001",
                "current_information_fraction": milestone,
            }

            result = await node._check_interim_trigger(experiment, None)

            assert result is not None
            assert result["triggered"] is True

    @pytest.mark.asyncio
    async def test_just_below_milestone_no_trigger(self, node):
        """Test no trigger when just below milestone."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.249,  # Just below 25%
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_above_100_percent_uses_75(self, node):
        """Test that above 100% still uses 75% milestone."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 1.2,  # Above 100%
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is not None
        assert result["milestone_reached"] == "75%"

    @pytest.mark.asyncio
    async def test_missing_information_fraction(self, node):
        """Test handling of missing information fraction."""
        experiment = {
            "experiment_id": "exp-001",
            # No current_information_fraction
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is None  # Defaults to 0


class TestCheckMilestoneAnalyzed:
    """Tests for _check_milestone_analyzed method."""

    @pytest.fixture
    def node(self):
        return InterimAnalyzerNode()

    @pytest.mark.asyncio
    async def test_not_analyzed_returns_false(self, node):
        """Test returns False when not analyzed."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_milestone_analyzed(mock_client, "exp-001", 0.50)

        assert result is False

    @pytest.mark.asyncio
    async def test_exact_match_returns_true(self, node):
        """Test returns True for exact milestone match."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"information_fraction": 0.50}]

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_milestone_analyzed(mock_client, "exp-001", 0.50)

        assert result is True

    @pytest.mark.asyncio
    async def test_within_tolerance_returns_true(self, node):
        """Test returns True for match within tolerance."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"information_fraction": 0.48}]  # Within 5% of 0.50

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_milestone_analyzed(mock_client, "exp-001", 0.50)

        assert result is True

    @pytest.mark.asyncio
    async def test_outside_tolerance_returns_false(self, node):
        """Test returns False for match outside tolerance."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"information_fraction": 0.25}]  # 25%, not 50%

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_milestone_analyzed(mock_client, "exp-001", 0.50)

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_exception(self, node):
        """Test returns False on exception."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_milestone_analyzed(mock_client, "exp-001", 0.50)

        assert result is False


class TestInterimTriggerStructure:
    """Tests for InterimTrigger structure."""

    @pytest.fixture
    def node(self):
        return InterimAnalyzerNode()

    @pytest.mark.asyncio
    async def test_trigger_has_required_fields(self, node):
        """Test that trigger has all required fields."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.50,
        }

        result = await node._check_interim_trigger(experiment, None)

        assert "experiment_id" in result
        assert "analysis_number" in result
        assert "information_fraction" in result
        assert "milestone_reached" in result
        assert "triggered" in result

    @pytest.mark.asyncio
    async def test_information_fraction_rounded(self, node):
        """Test that information fraction is rounded."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.501234567,
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result["information_fraction"] == 0.5012


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def node(self):
        return InterimAnalyzerNode()

    @pytest.mark.asyncio
    async def test_zero_information_fraction(self, node):
        """Test handling of zero information fraction."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": 0.0,
        }

        result = await node._check_interim_trigger(experiment, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_negative_information_fraction(self, node):
        """Test handling of negative information fraction."""
        experiment = {
            "experiment_id": "exp-001",
            "current_information_fraction": -0.1,
        }

        result = await node._check_interim_trigger(experiment, None)

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
                {"experiment_id": "exp-1", "current_information_fraction": 0.30},  # 25% trigger
                {"experiment_id": "exp-2", "current_information_fraction": 0.55},  # 50% trigger
                {"experiment_id": "exp-3", "current_information_fraction": 0.10},  # No trigger
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

            # exp-1 and exp-2 should trigger, exp-3 should not
            assert len(result["interim_triggers"]) == 2
            trigger_exp_ids = [t["experiment_id"] for t in result["interim_triggers"]]
            assert "exp-1" in trigger_exp_ids
            assert "exp-2" in trigger_exp_ids
            assert "exp-3" not in trigger_exp_ids

    @pytest.mark.asyncio
    async def test_missing_information_fraction_in_analysis(self, node):
        """Test handling of missing information_fraction in analysis record."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{}]  # Missing information_fraction

        mock_query = MagicMock()
        mock_query.select = MagicMock(return_value=mock_query)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.execute = AsyncMock(return_value=mock_result)
        mock_client.table = MagicMock(return_value=mock_query)

        result = await node._check_milestone_analyzed(mock_client, "exp-001", 0.50)

        # Should not match and return False
        assert result is False
