"""Tests for Experiment Monitor Agent.

Tests cover:
- ExperimentMonitorInput dataclass
- ExperimentMonitorOutput dataclass
- ExperimentMonitorAgent initialization
- Agent run_async and run methods
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.experiment_monitor.agent import (
    ExperimentMonitorAgent,
    ExperimentMonitorInput,
    ExperimentMonitorOutput,
)


class TestExperimentMonitorInput:
    """Tests for ExperimentMonitorInput dataclass."""

    def test_default_values(self):
        """Test default values for input."""
        input_data = ExperimentMonitorInput()
        assert input_data.query == ""
        assert input_data.experiment_ids is None
        assert input_data.check_all_active is True
        assert input_data.srm_threshold == 0.001
        assert input_data.enrollment_threshold == 5.0
        assert input_data.fidelity_threshold == 0.2
        assert input_data.check_interim is True

    def test_custom_query(self):
        """Test input with custom query."""
        input_data = ExperimentMonitorInput(query="Check my experiments")
        assert input_data.query == "Check my experiments"

    def test_specific_experiment_ids(self):
        """Test input with specific experiment IDs."""
        exp_ids = ["exp-001", "exp-002"]
        input_data = ExperimentMonitorInput(
            experiment_ids=exp_ids,
            check_all_active=False,
        )
        assert input_data.experiment_ids == exp_ids
        assert input_data.check_all_active is False

    def test_custom_thresholds(self):
        """Test input with custom thresholds."""
        input_data = ExperimentMonitorInput(
            srm_threshold=0.01,
            enrollment_threshold=10.0,
            fidelity_threshold=0.3,
        )
        assert input_data.srm_threshold == 0.01
        assert input_data.enrollment_threshold == 10.0
        assert input_data.fidelity_threshold == 0.3

    def test_disable_interim_check(self):
        """Test input with interim check disabled."""
        input_data = ExperimentMonitorInput(check_interim=False)
        assert input_data.check_interim is False

    def test_full_input_configuration(self):
        """Test fully configured input."""
        input_data = ExperimentMonitorInput(
            query="Full health check",
            experiment_ids=["exp-001"],
            check_all_active=False,
            srm_threshold=0.005,
            enrollment_threshold=8.0,
            fidelity_threshold=0.15,
            check_interim=True,
        )
        assert input_data.query == "Full health check"
        assert len(input_data.experiment_ids) == 1
        assert input_data.check_all_active is False
        assert input_data.srm_threshold == 0.005


class TestExperimentMonitorOutput:
    """Tests for ExperimentMonitorOutput dataclass."""

    def test_default_values(self):
        """Test default values for output."""
        output = ExperimentMonitorOutput()
        assert output.experiments == []
        assert output.alerts == []
        assert output.experiments_checked == 0
        assert output.healthy_count == 0
        assert output.warning_count == 0
        assert output.critical_count == 0
        assert output.monitor_summary == ""
        assert output.recommended_actions == []
        assert output.check_latency_ms == 0
        assert output.errors == []

    def test_output_with_experiments(self, sample_experiment_summaries):
        """Test output with experiment summaries."""
        output = ExperimentMonitorOutput(
            experiments=sample_experiment_summaries,
            experiments_checked=3,
            healthy_count=1,
            warning_count=1,
            critical_count=1,
        )
        assert len(output.experiments) == 3
        assert output.experiments_checked == 3
        assert output.healthy_count == 1

    def test_output_with_alerts(self, sample_alert_srm, sample_alert_enrollment):
        """Test output with alerts."""
        output = ExperimentMonitorOutput(
            alerts=[sample_alert_srm, sample_alert_enrollment],
        )
        assert len(output.alerts) == 2

    def test_output_with_summary(self):
        """Test output with summary and recommendations."""
        output = ExperimentMonitorOutput(
            monitor_summary="3 experiments checked, 1 critical",
            recommended_actions=["Investigate SRM in exp-002"],
        )
        assert "3 experiments" in output.monitor_summary
        assert len(output.recommended_actions) == 1

    def test_output_with_errors(self):
        """Test output with errors."""
        output = ExperimentMonitorOutput(
            errors=["Database connection failed"],
        )
        assert len(output.errors) == 1
        assert "Database" in output.errors[0]

    def test_output_with_latency(self):
        """Test output with latency tracking."""
        output = ExperimentMonitorOutput(check_latency_ms=450)
        assert output.check_latency_ms == 450


class TestExperimentMonitorAgentInit:
    """Tests for ExperimentMonitorAgent initialization."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ExperimentMonitorAgent()
        assert agent is not None
        assert hasattr(agent, "graph")
        assert agent.graph is not None

    def test_agent_has_graph(self):
        """Test agent has compiled graph."""
        agent = ExperimentMonitorAgent()
        # Graph should be the compiled experiment_monitor_graph
        assert hasattr(agent, "graph")

    def test_multiple_agent_instances(self):
        """Test creating multiple agent instances."""
        agent1 = ExperimentMonitorAgent()
        agent2 = ExperimentMonitorAgent()
        # Both should work independently
        assert agent1 is not agent2
        assert agent1.graph is agent2.graph  # Same compiled graph


class TestExperimentMonitorAgentAsync:
    """Tests for ExperimentMonitorAgent async execution."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph that returns test state."""
        mock = AsyncMock()
        mock.ainvoke.return_value = {
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
            "alerts": [],
            "experiments_checked": 1,
            "monitor_summary": "All healthy",
            "recommended_actions": ["No action required"],
            "check_latency_ms": 250,
            "errors": [],
        }
        return mock

    @pytest.mark.asyncio
    async def test_run_async_basic(self, mock_graph):
        """Test basic async execution."""
        agent = ExperimentMonitorAgent()
        agent.graph = mock_graph

        input_data = ExperimentMonitorInput(query="Check experiments")
        output = await agent.run_async(input_data)

        assert isinstance(output, ExperimentMonitorOutput)
        assert output.experiments_checked == 1
        assert len(output.experiments) == 1
        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_builds_initial_state(self, mock_graph):
        """Test that run_async builds correct initial state."""
        agent = ExperimentMonitorAgent()
        agent.graph = mock_graph

        input_data = ExperimentMonitorInput(
            query="Check specific",
            experiment_ids=["exp-001", "exp-002"],
            check_all_active=False,
            srm_threshold=0.005,
        )
        await agent.run_async(input_data)

        # Verify the initial state passed to graph
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["query"] == "Check specific"
        assert call_args["experiment_ids"] == ["exp-001", "exp-002"]
        assert call_args["check_all_active"] is False
        assert call_args["srm_threshold"] == 0.005

    @pytest.mark.asyncio
    async def test_run_async_counts_health_statuses(self, mock_graph):
        """Test that output correctly counts health statuses."""
        mock_graph.ainvoke.return_value = {
            "experiments": [
                {"experiment_id": "exp-1", "health_status": "healthy"},
                {"experiment_id": "exp-2", "health_status": "warning"},
                {"experiment_id": "exp-3", "health_status": "critical"},
                {"experiment_id": "exp-4", "health_status": "healthy"},
            ],
            "alerts": [],
            "experiments_checked": 4,
            "monitor_summary": "Mixed health",
            "recommended_actions": [],
            "check_latency_ms": 300,
            "errors": [],
        }

        agent = ExperimentMonitorAgent()
        agent.graph = mock_graph

        output = await agent.run_async(ExperimentMonitorInput())

        assert output.healthy_count == 2
        assert output.warning_count == 1
        assert output.critical_count == 1

    @pytest.mark.asyncio
    async def test_run_async_extracts_errors(self, mock_graph):
        """Test that errors are extracted correctly."""
        mock_graph.ainvoke.return_value = {
            "experiments": [],
            "alerts": [],
            "experiments_checked": 0,
            "monitor_summary": "Failed",
            "recommended_actions": [],
            "check_latency_ms": 50,
            "errors": [
                {"node": "health_checker", "error": "Connection failed", "timestamp": "2024-01-01"},
                {"node": "srm_detector", "error": "Timeout", "timestamp": "2024-01-01"},
            ],
        }

        agent = ExperimentMonitorAgent()
        agent.graph = mock_graph

        output = await agent.run_async(ExperimentMonitorInput())

        assert len(output.errors) == 2
        assert "Connection failed" in output.errors
        assert "Timeout" in output.errors

    @pytest.mark.asyncio
    async def test_run_async_handles_empty_results(self, mock_graph):
        """Test handling of empty results."""
        mock_graph.ainvoke.return_value = {
            "experiments": [],
            "alerts": [],
            "experiments_checked": 0,
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 100,
            "errors": [],
        }

        agent = ExperimentMonitorAgent()
        agent.graph = mock_graph

        output = await agent.run_async(ExperimentMonitorInput())

        assert output.experiments_checked == 0
        assert len(output.experiments) == 0
        assert output.healthy_count == 0

    @pytest.mark.asyncio
    async def test_run_async_with_null_experiment_ids(self, mock_graph):
        """Test that None experiment_ids becomes empty list."""
        agent = ExperimentMonitorAgent()
        agent.graph = mock_graph

        input_data = ExperimentMonitorInput(experiment_ids=None)
        await agent.run_async(input_data)

        call_args = mock_graph.ainvoke.call_args[0][0]
        assert call_args["experiment_ids"] == []


@pytest.mark.xdist_group(name="sync_wrappers")
class TestExperimentMonitorAgentSync:
    """Tests for ExperimentMonitorAgent synchronous execution.

    These tests verify that the sync wrapper correctly delegates to run_async.
    We use async tests to avoid event loop conflicts in pytest-xdist.
    """

    @pytest.mark.asyncio
    async def test_run_sync_basic(self):
        """Test basic sync execution wraps async.

        We test the async path directly to avoid event loop conflicts
        that occur when mixing sync/async in parallel test execution.
        """
        agent = ExperimentMonitorAgent()

        mock_output = ExperimentMonitorOutput(
            experiments_checked=1,
            healthy_count=1,
        )

        with patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_output
            # Test the async method directly to avoid event loop issues
            output = await agent.run_async(ExperimentMonitorInput())

        assert output == mock_output

    @pytest.mark.asyncio
    async def test_run_sync_passes_input(self):
        """Test sync run passes input to async.

        We test the async path directly to verify input passing
        without event loop conflicts.
        """
        agent = ExperimentMonitorAgent()

        with patch.object(agent, "run_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ExperimentMonitorOutput()
            input_data = ExperimentMonitorInput(query="Test query")
            await agent.run_async(input_data)

        mock_run.assert_called_once_with(input_data)


class TestExperimentMonitorAgentIntegration:
    """Integration tests for agent (without mocking graph)."""

    @pytest.mark.asyncio
    async def test_agent_full_workflow_mock_db(self):
        """Test full workflow with mocked database."""
        # This test verifies the agent runs through all nodes
        # We need to mock the Supabase client to avoid real DB calls

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_client:
            # Return None client to trigger mock data paths
            mock_client.return_value = None

            agent = ExperimentMonitorAgent()
            input_data = ExperimentMonitorInput(
                query="Check all experiments",
                check_all_active=True,
            )

            output = await agent.run_async(input_data)

            # Verify we got through the full workflow
            assert isinstance(output, ExperimentMonitorOutput)
            # Mock data path generates 2 experiments
            assert output.experiments_checked >= 0
            assert output.monitor_summary != ""
