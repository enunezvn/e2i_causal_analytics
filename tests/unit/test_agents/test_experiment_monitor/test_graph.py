"""Tests for Experiment Monitor Graph assembly.

Tests cover:
- Graph creation and compilation
- Node registration and ordering
- Edge connections
- Sequential workflow execution
- State propagation between nodes
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.experiment_monitor.graph import (
    create_experiment_monitor_graph,
    experiment_monitor_graph,
)


class TestGraphCreation:
    """Tests for graph creation function."""

    def test_create_graph_returns_compiled_graph(self):
        """Test that create_experiment_monitor_graph returns a compiled graph."""
        graph = create_experiment_monitor_graph()
        assert graph is not None

    def test_compiled_graph_is_singleton(self):
        """Test that experiment_monitor_graph is pre-compiled."""
        # The module-level graph should be ready to use
        assert experiment_monitor_graph is not None

    def test_graph_has_nodes(self):
        """Test that the graph has the expected nodes."""
        # We can inspect the graph's structure
        graph = create_experiment_monitor_graph()
        # LangGraph compiled graph has get_graph() method
        structure = graph.get_graph()

        # Get node names (nodes can be strings or objects with id attribute)
        node_ids = []
        for node in structure.nodes:
            if hasattr(node, "id"):
                node_ids.append(node.id)
            else:
                node_ids.append(str(node))

        assert "health_checker" in node_ids
        assert "srm_detector" in node_ids
        assert "interim_analyzer" in node_ids
        assert "fidelity_checker" in node_ids
        assert "alert_generator" in node_ids

    def test_graph_has_correct_entry_point(self):
        """Test that health_checker is the entry point."""
        graph = create_experiment_monitor_graph()
        structure = graph.get_graph()

        # Find edges from __start__
        start_edges = [e for e in structure.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "health_checker"

    def test_graph_has_sequential_edges(self):
        """Test that nodes are connected sequentially."""
        graph = create_experiment_monitor_graph()
        structure = graph.get_graph()

        # Build edge map
        edges = {e.source: e.target for e in structure.edges}

        # Verify sequential flow
        assert edges.get("health_checker") == "srm_detector"
        assert edges.get("srm_detector") == "interim_analyzer"
        assert edges.get("interim_analyzer") == "fidelity_checker"
        assert edges.get("fidelity_checker") == "alert_generator"
        assert edges.get("alert_generator") == "__end__"


class TestGraphExecution:
    """Tests for graph execution flow."""

    @pytest.fixture
    def mock_all_nodes(self):
        """Mock all node execution methods."""
        with (
            patch(
                "src.agents.experiment_monitor.nodes.HealthCheckerNode.execute",
                new_callable=AsyncMock,
            ) as mock_health,
            patch(
                "src.agents.experiment_monitor.nodes.SRMDetectorNode.execute",
                new_callable=AsyncMock,
            ) as mock_srm,
            patch(
                "src.agents.experiment_monitor.nodes.InterimAnalyzerNode.execute",
                new_callable=AsyncMock,
            ) as mock_interim,
            patch(
                "src.agents.experiment_monitor.nodes.AlertGeneratorNode.execute",
                new_callable=AsyncMock,
            ) as mock_alert,
        ):
            yield {
                "health": mock_health,
                "srm": mock_srm,
                "interim": mock_interim,
                "alert": mock_alert,
            }

    @pytest.mark.asyncio
    async def test_graph_executes_all_nodes_in_order(self, base_monitor_state):
        """Test that all nodes are executed in correct order."""
        execution_order = []

        async def track_health(state):
            execution_order.append("health_checker")
            state["status"] = "checking"
            return state

        async def track_srm(state):
            execution_order.append("srm_detector")
            return state

        async def track_interim(state):
            execution_order.append("interim_analyzer")
            return state

        async def track_alert(state):
            execution_order.append("alert_generator")
            state["status"] = "completed"
            return state

        with (
            patch("src.agents.experiment_monitor.graph.HealthCheckerNode") as MockHealth,
            patch("src.agents.experiment_monitor.graph.SRMDetectorNode") as MockSRM,
            patch("src.agents.experiment_monitor.graph.InterimAnalyzerNode") as MockInterim,
            patch("src.agents.experiment_monitor.graph.AlertGeneratorNode") as MockAlert,
        ):
            MockHealth.return_value.execute = track_health
            MockSRM.return_value.execute = track_srm
            MockInterim.return_value.execute = track_interim
            MockAlert.return_value.execute = track_alert

            graph = create_experiment_monitor_graph()
            await graph.ainvoke(base_monitor_state)

        assert execution_order == [
            "health_checker",
            "srm_detector",
            "interim_analyzer",
            "alert_generator",
        ]

    @pytest.mark.asyncio
    async def test_graph_state_propagation(self, base_monitor_state):
        """Test that state is propagated through nodes."""

        async def modify_health(state):
            state["experiments_checked"] = 5
            state["status"] = "checking"
            return state

        async def modify_srm(state):
            # Verify previous modification is visible
            assert state["experiments_checked"] == 5
            state["srm_issues"] = [{"experiment_id": "test", "detected": True}]
            return state

        async def modify_interim(state):
            # Verify both modifications are visible
            assert state["experiments_checked"] == 5
            assert len(state.get("srm_issues", [])) == 1
            state["interim_triggers"] = []
            return state

        async def modify_alert(state):
            # Verify all modifications are visible
            assert state["experiments_checked"] == 5
            assert len(state.get("srm_issues", [])) == 1
            state["status"] = "completed"
            state["monitor_summary"] = "Test complete"
            return state

        with (
            patch("src.agents.experiment_monitor.graph.HealthCheckerNode") as MockHealth,
            patch("src.agents.experiment_monitor.graph.SRMDetectorNode") as MockSRM,
            patch("src.agents.experiment_monitor.graph.InterimAnalyzerNode") as MockInterim,
            patch("src.agents.experiment_monitor.graph.AlertGeneratorNode") as MockAlert,
        ):
            MockHealth.return_value.execute = modify_health
            MockSRM.return_value.execute = modify_srm
            MockInterim.return_value.execute = modify_interim
            MockAlert.return_value.execute = modify_alert

            graph = create_experiment_monitor_graph()
            final_state = await graph.ainvoke(base_monitor_state)

        assert final_state["status"] == "completed"
        assert final_state["experiments_checked"] == 5
        assert final_state["monitor_summary"] == "Test complete"


class TestNodeCount:
    """Tests for node count in the graph."""

    def test_graph_has_five_nodes(self):
        """Test that graph has exactly 5 processing nodes."""
        graph = create_experiment_monitor_graph()
        structure = graph.get_graph()

        # Get node names
        node_names = []
        for node in structure.nodes:
            if hasattr(node, "id"):
                node_names.append(node.id)
            else:
                node_names.append(str(node))

        # Filter out __start__ and __end__
        processing_nodes = [n for n in node_names if n not in ["__start__", "__end__"]]
        assert len(processing_nodes) == 5

    def test_graph_has_six_edges(self):
        """Test that graph has exactly 6 edges (including start/end)."""
        graph = create_experiment_monitor_graph()
        structure = graph.get_graph()

        # __start__ -> health_checker -> srm_detector -> interim_analyzer -> fidelity_checker -> alert_generator -> __end__
        assert len(structure.edges) == 6


class TestGraphWithRealNodes:
    """Tests that verify graph works with real node instances."""

    @pytest.mark.asyncio
    async def test_graph_handles_node_errors_gracefully(self, base_monitor_state):
        """Test that errors in nodes are handled."""
        # Patch the Supabase client to return None (triggering mock data)
        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_client:
            mock_client.return_value = None

            graph = experiment_monitor_graph
            final_state = await graph.ainvoke(base_monitor_state)

            # Graph should complete even with mock data
            assert final_state["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_graph_returns_valid_final_state(self, base_monitor_state):
        """Test that graph returns a valid final state."""
        with patch(
            "src.memory.services.factories.get_supabase_client",
            new_callable=AsyncMock,
        ) as mock_client:
            mock_client.return_value = None

            final_state = await experiment_monitor_graph.ainvoke(base_monitor_state)

            # Verify final state has expected fields
            assert "experiments" in final_state
            assert "srm_issues" in final_state
            assert "interim_triggers" in final_state
            assert "alerts" in final_state
            assert "status" in final_state
            assert "monitor_summary" in final_state
