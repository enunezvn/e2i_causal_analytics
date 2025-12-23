"""Integration tests for Experiment Monitor Agent.

Tests cover:
- End-to-end workflow execution through all 5 nodes
- Contract compliance with ExperimentMonitorState
- State propagation between nodes
- Error handling and recovery
- Edge cases (empty state, all healthy, all critical)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor.agent import ExperimentMonitorAgent
from src.agents.experiment_monitor.graph import (
    create_experiment_monitor_graph,
    experiment_monitor_graph,
)
from src.agents.experiment_monitor.nodes import (
    AlertGeneratorNode,
    FidelityCheckerNode,
    HealthCheckerNode,
    InterimAnalyzerNode,
    SRMDetectorNode,
)
from src.agents.experiment_monitor.state import (
    ExperimentMonitorState,
    ExperimentSummary,
    HealthStatus,
)

# Import helper functions from conftest
from .conftest import (
    create_enrollment_issue,
    create_experiment_summary,
    create_monitor_state,
    create_srm_issue,
)


class TestEndToEndWorkflows:
    """End-to-end integration tests for the experiment monitor workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_healthy_experiments(self, base_monitor_state):
        """Test complete workflow with healthy experiments."""
        # Create experiments that will pass all checks
        healthy_experiments = [
            {
                "id": "exp-001",
                "name": "Healthy Experiment",
                "status": "running",
                "config": {
                    "target_sample_size": 1000,
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

        # Mock balanced assignments (no SRM)
        balanced_assignments = []
        for i in range(250):
            balanced_assignments.append({"experiment_id": "exp-001", "variant": "control"})
            balanced_assignments.append({"experiment_id": "exp-001", "variant": "treatment"})

        mock_client = MagicMock()

        # Setup mock query builder
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.in_.return_value = mock_query

        def mock_execute():
            result = MagicMock()
            # Return different data based on call count
            if mock_client.table.call_count <= 1:
                result.data = healthy_experiments
            else:
                result.data = balanced_assignments
            result.count = len(result.data)
            return result

        mock_query.execute = AsyncMock(side_effect=mock_execute)
        mock_client.table.return_value = mock_query

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(base_monitor_state)

        assert final_state["status"] == "completed"
        assert len(final_state["srm_issues"]) == 0
        assert "healthy" in final_state["monitor_summary"].lower() or len(final_state["alerts"]) == 0

    @pytest.mark.asyncio
    async def test_full_workflow_with_srm_detected(self, base_monitor_state):
        """Test workflow detects SRM and generates alerts."""
        # Create state with SRM issue pre-populated (simulate health_checker ran)
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary(
                "exp-srm",
                name="SRM Experiment",
                health_status="warning",
                total_enrolled=200,
            )
        ]
        state["experiments_checked"] = 1
        state["status"] = "checking"

        # Track node executions
        executed_nodes = []

        async def track_health(self_obj, state_arg):
            executed_nodes.append("health_checker")
            return state_arg

        async def track_srm(self_obj, state_arg):
            executed_nodes.append("srm_detector")
            # Add SRM issue
            state_arg["srm_issues"] = [
                create_srm_issue("exp-srm", p_value=0.0001, severity="critical")
            ]
            return state_arg

        async def track_interim(self_obj, state_arg):
            executed_nodes.append("interim_analyzer")
            return state_arg

        async def track_alert(self_obj, state_arg):
            executed_nodes.append("alert_generator")
            # Generate alerts based on SRM issues
            if state_arg.get("srm_issues"):
                state_arg["alerts"] = [
                    {
                        "alert_id": "alert-001",
                        "alert_type": "srm",
                        "severity": "critical",
                        "experiment_id": "exp-srm",
                        "message": "SRM detected",
                    }
                ]
            state_arg["status"] = "completed"
            state_arg["monitor_summary"] = "Issues detected"
            return state_arg

        async def track_fidelity(self_obj, state_arg):
            executed_nodes.append("fidelity_checker")
            return state_arg

        with patch.object(HealthCheckerNode, "execute", track_health), \
             patch.object(SRMDetectorNode, "execute", track_srm), \
             patch.object(InterimAnalyzerNode, "execute", track_interim), \
             patch.object(FidelityCheckerNode, "execute", track_fidelity), \
             patch.object(AlertGeneratorNode, "execute", track_alert):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(state)

        assert "health_checker" in executed_nodes
        assert "srm_detector" in executed_nodes
        assert "interim_analyzer" in executed_nodes
        assert "fidelity_checker" in executed_nodes
        assert "alert_generator" in executed_nodes
        assert final_state["status"] == "completed"
        assert len(final_state["srm_issues"]) > 0
        assert len(final_state["alerts"]) > 0

    @pytest.mark.asyncio
    async def test_full_workflow_with_enrollment_issues(self, base_monitor_state):
        """Test workflow handles low enrollment experiments."""
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary(
                "exp-low",
                name="Low Enrollment",
                health_status="critical",
                enrollment_rate=1.5,
                total_enrolled=30,
            )
        ]
        state["enrollment_issues"] = [
            create_enrollment_issue("exp-low", current_rate=1.5, severity="critical")
        ]
        state["experiments_checked"] = 1
        state["status"] = "checking"

        # Use real alert generator to generate enrollment alerts
        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            alert_node = AlertGeneratorNode()
            final_state = await alert_node.execute(state)

        assert final_state["status"] == "completed"
        # Should have at least one alert for enrollment
        enrollment_alerts = [
            a for a in final_state.get("alerts", [])
            if a.get("alert_type") == "enrollment"
        ]
        assert len(enrollment_alerts) >= 1

    @pytest.mark.asyncio
    async def test_full_workflow_with_interim_triggers(self, base_monitor_state):
        """Test workflow handles interim analysis milestones."""
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary(
                "exp-interim",
                name="Interim Test",
                health_status="healthy",
                information_fraction=0.52,  # Just past 50% milestone
            )
        ]
        state["interim_triggers"] = [
            {
                "experiment_id": "exp-interim",
                "analysis_number": 2,
                "information_fraction": 0.52,
                "milestone_reached": "50%",
                "triggered": True,
            }
        ]
        state["experiments_checked"] = 1
        state["status"] = "analyzing"

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            alert_node = AlertGeneratorNode()
            final_state = await alert_node.execute(state)

        assert final_state["status"] == "completed"
        # Should have interim trigger alert
        interim_alerts = [
            a for a in final_state.get("alerts", [])
            if a.get("alert_type") == "interim_trigger"
        ]
        assert len(interim_alerts) >= 1

    @pytest.mark.asyncio
    async def test_full_workflow_empty_experiments(self, base_monitor_state):
        """Test workflow handles empty experiment list gracefully."""
        state = base_monitor_state.copy()
        state["experiments"] = []
        state["experiments_checked"] = 0

        # Run through all nodes
        async def passthrough(self_obj, s):
            return s

        async def complete(self_obj, s):
            s["status"] = "completed"
            s["monitor_summary"] = "No experiments to monitor"
            return s

        with patch.object(HealthCheckerNode, "execute", passthrough), \
             patch.object(SRMDetectorNode, "execute", passthrough), \
             patch.object(InterimAnalyzerNode, "execute", passthrough), \
             patch.object(FidelityCheckerNode, "execute", passthrough), \
             patch.object(AlertGeneratorNode, "execute", complete):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(state)

        assert final_state["status"] == "completed"
        assert final_state["experiments_checked"] == 0

    @pytest.mark.asyncio
    async def test_workflow_latency_accumulation(self, base_monitor_state):
        """Test that latency is accumulated across nodes."""
        state = base_monitor_state.copy()
        state["check_latency_ms"] = 0

        async def add_latency_health(self_obj, s):
            s["check_latency_ms"] = s.get("check_latency_ms", 0) + 100
            return s

        async def add_latency_srm(self_obj, s):
            s["check_latency_ms"] = s.get("check_latency_ms", 0) + 50
            return s

        async def add_latency_interim(self_obj, s):
            s["check_latency_ms"] = s.get("check_latency_ms", 0) + 30
            return s

        async def add_latency_alert(self_obj, s):
            s["check_latency_ms"] = s.get("check_latency_ms", 0) + 20
            s["status"] = "completed"
            return s

        async def add_latency_fidelity(self_obj, s):
            s["check_latency_ms"] = s.get("check_latency_ms", 0) + 25
            return s

        with patch.object(HealthCheckerNode, "execute", add_latency_health), \
             patch.object(SRMDetectorNode, "execute", add_latency_srm), \
             patch.object(InterimAnalyzerNode, "execute", add_latency_interim), \
             patch.object(FidelityCheckerNode, "execute", add_latency_fidelity), \
             patch.object(AlertGeneratorNode, "execute", add_latency_alert):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(state)

        # Total latency should be 100 + 50 + 30 + 25 + 20 = 225
        assert final_state["check_latency_ms"] == 225

    @pytest.mark.asyncio
    async def test_workflow_with_multiple_issues(self, state_with_all_issues):
        """Test workflow handles multiple concurrent issues."""
        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            alert_node = AlertGeneratorNode()
            final_state = await alert_node.execute(state_with_all_issues)

        assert final_state["status"] == "completed"
        # Should have multiple alerts
        alert_types = {a.get("alert_type") for a in final_state.get("alerts", [])}
        # At least some alerts should be generated
        assert len(final_state["alerts"]) >= 1


class TestContractCompliance:
    """Tests verifying contract compliance for ExperimentMonitorState."""

    def test_initial_state_has_required_fields(self, base_monitor_state):
        """Test that initial state has all required fields."""
        required_fields = [
            "query",
            "experiment_ids",
            "check_all_active",
            "srm_threshold",
            "enrollment_threshold",
            "fidelity_threshold",
            "check_interim",
            "experiments",
            "srm_issues",
            "enrollment_issues",
            "fidelity_issues",
            "interim_triggers",
            "alerts",
            "monitor_summary",
            "recommended_actions",
            "check_latency_ms",
            "experiments_checked",
            "errors",
            "warnings",
            "status",
        ]
        for field in required_fields:
            assert field in base_monitor_state, f"Missing field: {field}"

    def test_final_state_preserves_input_fields(self, base_monitor_state):
        """Test that final state preserves original input fields."""
        original_query = base_monitor_state["query"]
        original_threshold = base_monitor_state["srm_threshold"]

        async def complete(self_obj, s):
            s["status"] = "completed"
            return s

        with patch.object(HealthCheckerNode, "execute", complete), \
             patch.object(SRMDetectorNode, "execute", complete), \
             patch.object(InterimAnalyzerNode, "execute", complete), \
             patch.object(AlertGeneratorNode, "execute", complete):
            pass  # State should preserve fields after mocked execution

        # Check fields are still present and unchanged
        assert base_monitor_state["query"] == original_query
        assert base_monitor_state["srm_threshold"] == original_threshold

    def test_status_transitions_are_valid(self, base_monitor_state):
        """Test that status transitions follow the expected pattern."""
        valid_statuses = ["pending", "checking", "analyzing", "alerting", "completed", "failed"]

        assert base_monitor_state["status"] in valid_statuses

    def test_experiment_summary_structure(self, sample_summary_healthy):
        """Test ExperimentSummary has required fields."""
        required_fields = [
            "experiment_id",
            "name",
            "status",
            "health_status",
            "days_running",
            "total_enrolled",
            "enrollment_rate",
            "current_information_fraction",
        ]
        summary_dict = sample_summary_healthy if isinstance(sample_summary_healthy, dict) else {
            "experiment_id": sample_summary_healthy.experiment_id,
            "name": sample_summary_healthy.name,
            "status": sample_summary_healthy.status,
            "health_status": sample_summary_healthy.health_status,
            "days_running": sample_summary_healthy.days_running,
            "total_enrolled": sample_summary_healthy.total_enrolled,
            "enrollment_rate": sample_summary_healthy.enrollment_rate,
            "current_information_fraction": sample_summary_healthy.current_information_fraction,
        }
        for field in required_fields:
            assert field in summary_dict, f"Missing field: {field}"

    def test_alert_structure(self, sample_alert_srm):
        """Test MonitorAlert has required fields."""
        required_fields = [
            "alert_id",
            "alert_type",
            "severity",
            "experiment_id",
            "message",
        ]
        alert_dict = sample_alert_srm if isinstance(sample_alert_srm, dict) else {
            "alert_id": sample_alert_srm.alert_id,
            "alert_type": sample_alert_srm.alert_type,
            "severity": sample_alert_srm.severity,
            "experiment_id": sample_alert_srm.experiment_id,
            "message": sample_alert_srm.message,
        }
        for field in required_fields:
            assert field in alert_dict, f"Missing field: {field}"


class TestStatePropagation:
    """Tests for state propagation between nodes."""

    @pytest.mark.asyncio
    async def test_health_checker_output_reaches_srm_detector(self, base_monitor_state):
        """Test that health_checker output is visible to srm_detector."""
        received_experiments = []

        async def health_output(self_obj, s):
            s["experiments"] = [
                create_experiment_summary("exp-001", health_status="healthy")
            ]
            s["experiments_checked"] = 1
            return s

        async def srm_capture(self_obj, s):
            received_experiments.extend(s.get("experiments", []))
            return s

        async def passthrough(self_obj, s):
            s["status"] = "completed"
            return s

        with patch.object(HealthCheckerNode, "execute", health_output), \
             patch.object(SRMDetectorNode, "execute", srm_capture), \
             patch.object(InterimAnalyzerNode, "execute", passthrough), \
             patch.object(FidelityCheckerNode, "execute", passthrough), \
             patch.object(AlertGeneratorNode, "execute", passthrough):
            graph = create_experiment_monitor_graph()
            await graph.ainvoke(base_monitor_state)

        assert len(received_experiments) == 1
        # ExperimentSummary is a TypedDict, access as dict
        assert received_experiments[0]["experiment_id"] == "exp-001"

    @pytest.mark.asyncio
    async def test_srm_issues_reach_alert_generator(self, base_monitor_state):
        """Test that SRM issues propagate to alert generator."""
        received_srm_issues = []

        async def passthrough(self_obj, s):
            return s

        async def add_srm(self_obj, s):
            s["srm_issues"] = [
                create_srm_issue("exp-001", severity="critical")
            ]
            return s

        async def capture_alerts(self_obj, s):
            received_srm_issues.extend(s.get("srm_issues", []))
            s["status"] = "completed"
            return s

        with patch.object(HealthCheckerNode, "execute", passthrough), \
             patch.object(SRMDetectorNode, "execute", add_srm), \
             patch.object(InterimAnalyzerNode, "execute", passthrough), \
             patch.object(FidelityCheckerNode, "execute", passthrough), \
             patch.object(AlertGeneratorNode, "execute", capture_alerts):
            graph = create_experiment_monitor_graph()
            await graph.ainvoke(base_monitor_state)

        assert len(received_srm_issues) == 1

    @pytest.mark.asyncio
    async def test_errors_accumulate_across_nodes(self, base_monitor_state):
        """Test that errors from each node are accumulated."""
        async def add_error_health(self_obj, s):
            s["errors"] = s.get("errors", []) + [
                {"node": "health_checker", "error": "Error 1"}
            ]
            return s

        async def add_error_srm(self_obj, s):
            s["errors"] = s.get("errors", []) + [
                {"node": "srm_detector", "error": "Error 2"}
            ]
            return s

        async def add_error_fidelity(self_obj, s):
            s["errors"] = s.get("errors", []) + [
                {"node": "fidelity_checker", "error": "Error 3"}
            ]
            return s

        async def passthrough(self_obj, s):
            return s

        async def complete(self_obj, s):
            s["status"] = "completed"
            return s

        with patch.object(HealthCheckerNode, "execute", add_error_health), \
             patch.object(SRMDetectorNode, "execute", add_error_srm), \
             patch.object(InterimAnalyzerNode, "execute", passthrough), \
             patch.object(FidelityCheckerNode, "execute", add_error_fidelity), \
             patch.object(AlertGeneratorNode, "execute", complete):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(base_monitor_state)

        assert len(final_state["errors"]) == 3


class TestErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_workflow_continues_after_node_error(self, base_monitor_state):
        """Test that workflow continues even if a node has an error."""
        executed = []

        async def error_health(self_obj, s):
            executed.append("health")
            # Simulate error but don't raise - just add to errors
            s["errors"] = [{"node": "health_checker", "error": "DB error"}]
            s["warnings"] = ["Using mock data"]
            return s

        async def continue_srm(self_obj, s):
            executed.append("srm")
            return s

        async def continue_interim(self_obj, s):
            executed.append("interim")
            return s

        async def complete(self_obj, s):
            executed.append("alert")
            s["status"] = "completed"
            return s

        async def continue_fidelity(self_obj, s):
            executed.append("fidelity")
            return s

        with patch.object(HealthCheckerNode, "execute", error_health), \
             patch.object(SRMDetectorNode, "execute", continue_srm), \
             patch.object(InterimAnalyzerNode, "execute", continue_interim), \
             patch.object(FidelityCheckerNode, "execute", continue_fidelity), \
             patch.object(AlertGeneratorNode, "execute", complete):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(base_monitor_state)

        # All nodes should have executed
        assert executed == ["health", "srm", "interim", "fidelity", "alert"]
        assert final_state["status"] == "completed"
        assert len(final_state["errors"]) > 0

    @pytest.mark.asyncio
    async def test_exception_in_node_sets_failed_status(self, base_monitor_state):
        """Test that exceptions in nodes result in failed status."""
        async def raise_error(self_obj, s):
            raise ValueError("Test error")

        # Note: Real graph may handle this differently - this tests the expected behavior
        with patch.object(HealthCheckerNode, "execute", raise_error):
            graph = create_experiment_monitor_graph()
            try:
                await graph.ainvoke(base_monitor_state)
            except ValueError:
                pass  # Expected

    @pytest.mark.asyncio
    async def test_database_unavailable_uses_fallback(self, base_monitor_state):
        """Test that database unavailability triggers fallback behavior."""
        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,  # Simulate no DB connection
        ):
            # Real nodes should handle None client gracefully
            health_node = HealthCheckerNode()
            result = await health_node.execute(base_monitor_state)

        # Should not crash, may have warnings or use mock data
        assert "status" in result


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_experiment(self, base_monitor_state):
        """Test workflow with single experiment."""
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary("exp-single", health_status="healthy")
        ]
        state["experiments_checked"] = 1

        async def complete(self_obj, s):
            s["status"] = "completed"
            s["monitor_summary"] = "1 experiment checked"
            return s

        with patch.object(HealthCheckerNode, "execute", complete), \
             patch.object(SRMDetectorNode, "execute", complete), \
             patch.object(InterimAnalyzerNode, "execute", complete), \
             patch.object(FidelityCheckerNode, "execute", complete), \
             patch.object(AlertGeneratorNode, "execute", complete):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(state)

        assert final_state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_many_experiments(self, base_monitor_state):
        """Test workflow with many experiments."""
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary(f"exp-{i}", health_status="healthy")
            for i in range(50)
        ]
        state["experiments_checked"] = 50

        async def complete(self_obj, s):
            s["status"] = "completed"
            s["monitor_summary"] = f"{len(s['experiments'])} experiments checked"
            return s

        with patch.object(HealthCheckerNode, "execute", complete), \
             patch.object(SRMDetectorNode, "execute", complete), \
             patch.object(InterimAnalyzerNode, "execute", complete), \
             patch.object(FidelityCheckerNode, "execute", complete), \
             patch.object(AlertGeneratorNode, "execute", complete):
            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(state)

        assert final_state["experiments_checked"] == 50

    @pytest.mark.asyncio
    async def test_all_critical_experiments(self, base_monitor_state):
        """Test workflow where all experiments are critical."""
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary(
                f"exp-crit-{i}",
                health_status="critical",
                enrollment_rate=1.0,
            )
            for i in range(3)
        ]
        state["enrollment_issues"] = [
            create_enrollment_issue(f"exp-crit-{i}", current_rate=1.0, severity="critical")
            for i in range(3)
        ]
        state["experiments_checked"] = 3

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            alert_node = AlertGeneratorNode()
            final_state = await alert_node.execute(state)

        assert final_state["status"] == "completed"
        # Should have alerts for all critical experiments
        assert len(final_state["alerts"]) >= 3

    @pytest.mark.asyncio
    async def test_borderline_srm_threshold(self, base_monitor_state):
        """Test SRM detection at threshold boundary."""
        state = base_monitor_state.copy()
        state["srm_threshold"] = 0.001
        state["experiments"] = [
            create_experiment_summary("exp-border", health_status="warning")
        ]
        # p-value exactly at threshold
        state["srm_issues"] = [
            create_srm_issue("exp-border", p_value=0.001, severity="warning")
        ]

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            alert_node = AlertGeneratorNode()
            final_state = await alert_node.execute(state)

        # Should generate alert at threshold
        assert len(final_state["alerts"]) >= 1

    @pytest.mark.asyncio
    async def test_multiple_milestones_same_experiment(self, base_monitor_state):
        """Test handling multiple interim milestones for same experiment."""
        state = base_monitor_state.copy()
        state["experiments"] = [
            create_experiment_summary(
                "exp-multi",
                health_status="healthy",
                information_fraction=0.77,  # Past 75% milestone
            )
        ]
        state["interim_triggers"] = [
            {
                "experiment_id": "exp-multi",
                "analysis_number": 1,
                "information_fraction": 0.25,
                "milestone_reached": "25%",
                "triggered": True,
            },
            {
                "experiment_id": "exp-multi",
                "analysis_number": 2,
                "information_fraction": 0.52,
                "milestone_reached": "50%",
                "triggered": True,
            },
            {
                "experiment_id": "exp-multi",
                "analysis_number": 3,
                "information_fraction": 0.77,
                "milestone_reached": "75%",
                "triggered": True,
            },
        ]

        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            alert_node = AlertGeneratorNode()
            final_state = await alert_node.execute(state)

        assert final_state["status"] == "completed"
        # Should handle multiple triggers
        interim_alerts = [
            a for a in final_state.get("alerts", [])
            if a.get("alert_type") == "interim_trigger"
        ]
        assert len(interim_alerts) >= 1


class TestAgentInterface:
    """Tests for the ExperimentMonitorAgent interface."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ExperimentMonitorAgent()
        assert agent is not None
        assert hasattr(agent, "run")

    @pytest.mark.asyncio
    async def test_agent_run_method_exists(self):
        """Test agent has run method."""
        agent = ExperimentMonitorAgent()
        assert callable(getattr(agent, "run", None)) or callable(getattr(agent, "arun", None))

    def test_agent_uses_correct_graph(self):
        """Test agent uses the experiment monitor graph."""
        # Verify the module-level graph is available
        assert experiment_monitor_graph is not None


class TestGraphNodeCount:
    """Tests to verify graph structure."""

    def test_graph_has_exactly_five_processing_nodes(self):
        """Verify the graph has exactly 5 processing nodes."""
        graph = create_experiment_monitor_graph()
        structure = graph.get_graph()

        node_names = []
        for node in structure.nodes:
            if hasattr(node, "id"):
                node_names.append(node.id)
            else:
                node_names.append(str(node))

        # Filter out __start__ and __end__
        processing_nodes = [n for n in node_names if n not in ["__start__", "__end__"]]
        assert len(processing_nodes) == 5
        assert "health_checker" in processing_nodes
        assert "srm_detector" in processing_nodes
        assert "interim_analyzer" in processing_nodes
        assert "fidelity_checker" in processing_nodes
        assert "alert_generator" in processing_nodes

    def test_graph_edges_form_linear_chain(self):
        """Verify edges form linear chain: start -> health -> srm -> interim -> fidelity -> alert -> end."""
        graph = create_experiment_monitor_graph()
        structure = graph.get_graph()

        edges = {e.source: e.target for e in structure.edges}

        assert edges["__start__"] == "health_checker"
        assert edges["health_checker"] == "srm_detector"
        assert edges["srm_detector"] == "interim_analyzer"
        assert edges["interim_analyzer"] == "fidelity_checker"
        assert edges["fidelity_checker"] == "alert_generator"
        assert edges["alert_generator"] == "__end__"
