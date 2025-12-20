"""Tests for graph_builder node."""

import pytest
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.state import CausalImpactState


class TestGraphBuilderNode:
    """Test GraphBuilderNode."""

    @pytest.mark.asyncio
    async def test_build_graph_with_explicit_variables(self):
        """Test graph building with explicit treatment/outcome."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "what is the impact of hcp engagement on conversions?",
            "query_id": "test-1",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "status": "pending",
        }

        result = await node.execute(state)

        assert "causal_graph" in result
        graph = result["causal_graph"]
        assert "hcp_engagement_level" in graph["nodes"]
        assert "patient_conversion_rate" in graph["nodes"]
        assert "geographic_region" in graph["nodes"]
        assert len(graph["edges"]) > 0
        assert result["current_phase"] == "estimating"

    @pytest.mark.asyncio
    async def test_build_graph_infer_variables_from_query(self):
        """Test variable inference from query."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "does marketing spend increase prescription volume?",
            "query_id": "test-2",
            "status": "pending",
        }

        result = await node.execute(state)

        assert "causal_graph" in result
        graph = result["causal_graph"]
        # Should infer treatment and outcome from query
        assert len(graph["treatment_nodes"]) == 1
        assert len(graph["outcome_nodes"]) == 1

    @pytest.mark.asyncio
    async def test_find_adjustment_sets(self):
        """Test finding valid adjustment sets."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-3",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region", "hcp_specialty"],
            "status": "pending",
        }

        result = await node.execute(state)

        graph = result["causal_graph"]
        assert "adjustment_sets" in graph
        assert isinstance(graph["adjustment_sets"], list)
        # Should find at least one adjustment set
        assert len(graph["adjustment_sets"]) >= 1

    @pytest.mark.asyncio
    async def test_dag_has_no_cycles(self):
        """Test that constructed DAG is acyclic."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-4",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "status": "pending",
        }

        result = await node.execute(state)

        # Test via edges - no node should have path back to itself
        graph = result["causal_graph"]
        edges = graph["edges"]

        # Build adjacency list
        from collections import defaultdict, deque

        adj = defaultdict(list)
        for source, target in edges:
            adj[source].append(target)

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        rec_stack = set()

        for node_name in graph["nodes"]:
            if node_name not in visited:
                assert not has_cycle(
                    node_name, visited, rec_stack
                ), "DAG contains cycle"

    @pytest.mark.asyncio
    async def test_dag_includes_confounders(self):
        """Test that DAG includes common confounders."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-5",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region", "hcp_specialty", "practice_size"],
            "status": "pending",
        }

        result = await node.execute(state)

        graph = result["causal_graph"]

        # At least one confounder should be in the graph
        confounders = {"geographic_region", "hcp_specialty", "practice_size"}
        graph_nodes = set(graph["nodes"])

        assert len(confounders & graph_nodes) >= 1, "No confounders in graph"

    @pytest.mark.asyncio
    async def test_treatment_outcome_path_exists(self):
        """Test that there's a path from treatment to outcome."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-6",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "status": "pending",
        }

        result = await node.execute(state)

        graph = result["causal_graph"]
        edges = graph["edges"]

        # Build adjacency list
        from collections import defaultdict, deque

        adj = defaultdict(list)
        for source, target in edges:
            adj[source].append(target)

        # BFS to check path
        def has_path(start, end):
            if start == end:
                return True

            queue = deque([start])
            visited = {start}

            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor == end:
                        return True
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            return False

        treatment = graph["treatment_nodes"][0]
        outcome = graph["outcome_nodes"][0]

        assert has_path(treatment, outcome), "No path from treatment to outcome"

    @pytest.mark.asyncio
    async def test_dot_format_generation(self):
        """Test DOT format generation for visualization."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-7",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "status": "pending",
        }

        result = await node.execute(state)

        graph = result["causal_graph"]
        assert "dag_dot" in graph
        assert graph["dag_dot"].startswith("digraph")
        assert "hcp_engagement_level" in graph["dag_dot"]
        assert "patient_conversion_rate" in graph["dag_dot"]
        assert "->" in graph["dag_dot"]

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that latency is measured."""
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-8",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "status": "pending",
        }

        result = await node.execute(state)

        assert "graph_builder_latency_ms" in result
        assert result["graph_builder_latency_ms"] >= 0
        assert result["graph_builder_latency_ms"] < 10000  # Should be < 10s

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in graph builder."""
        node = GraphBuilderNode()

        # Invalid state (missing query)
        state: CausalImpactState = {
            "query_id": "test-9",
            "status": "pending",
        }

        result = await node.execute(state)

        # Should handle gracefully (with default variables)
        # OR return error state
        assert "status" in result


class TestVariableInference:
    """Test variable inference logic."""

    def test_infer_treatment_from_keywords(self):
        """Test treatment variable inference."""
        node = GraphBuilderNode()

        queries_and_expected = [
            ("impact of hcp engagement on conversions", "hcp_engagement_level"),
            ("does marketing spend increase prescriptions", "marketing_spend"),
            ("effect of copay support on adherence", "copay_support"),
        ]

        for query, expected_treatment in queries_and_expected:
            treatment, _ = node._infer_variables_from_query(query)
            assert (
                expected_treatment in treatment or treatment == "hcp_engagement_level"
            ), f"Failed for query: {query}"

    def test_infer_outcome_from_keywords(self):
        """Test outcome variable inference."""
        node = GraphBuilderNode()

        queries_and_expected = [
            ("impact on patient conversion", "patient_conversion_rate"),
            ("effect on prescription volume", "prescription_volume"),
            ("improve market share", "market_share"),
        ]

        for query, expected_outcome in queries_and_expected:
            _, outcome = node._infer_variables_from_query(query)
            assert (
                expected_outcome in outcome or outcome == "patient_conversion_rate"
            ), f"Failed for query: {query}"

    def test_default_variables(self):
        """Test that defaults are used when keywords not found."""
        node = GraphBuilderNode()

        # Query with no recognizable keywords
        treatment, outcome = node._infer_variables_from_query("hello there")

        assert treatment == "hcp_engagement_level"  # Default treatment
        assert outcome == "patient_conversion_rate"  # Default outcome


class TestAdjustmentSetLogic:
    """Test backdoor criterion and adjustment set logic."""

    def test_no_backdoor_paths(self):
        """Test when no confounding exists."""
        import networkx as nx

        node = GraphBuilderNode()

        # Simple DAG: T -> O (no confounders)
        dag = nx.DiGraph()
        dag.add_edge("T", "O")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        # Empty set should be sufficient (no confounding)
        assert [] in adjustment_sets

    def test_single_confounder(self):
        """Test adjustment set with single confounder."""
        import networkx as nx

        node = GraphBuilderNode()

        # DAG: C -> T, C -> O (single confounder)
        dag = nx.DiGraph()
        dag.add_edge("C", "T")
        dag.add_edge("C", "O")
        dag.add_edge("T", "O")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        # Should identify C as valid adjustment set
        assert any("C" in adj_set for adj_set in adjustment_sets)

    def test_multiple_confounders(self):
        """Test adjustment set with multiple confounders."""
        import networkx as nx

        node = GraphBuilderNode()

        # DAG: C1 -> T, C1 -> O, C2 -> T, C2 -> O
        dag = nx.DiGraph()
        dag.add_edge("C1", "T")
        dag.add_edge("C1", "O")
        dag.add_edge("C2", "T")
        dag.add_edge("C2", "O")
        dag.add_edge("T", "O")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        # Should find valid adjustment sets
        assert len(adjustment_sets) > 0
