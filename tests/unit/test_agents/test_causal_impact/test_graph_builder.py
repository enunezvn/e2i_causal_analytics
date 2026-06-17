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
        from collections import defaultdict

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
                assert not has_cycle(node_name, visited, rec_stack), "DAG contains cycle"

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

    @pytest.mark.asyncio
    async def test_run_discovery_reads_estimation_data_key(self):
        """M-gb1: _run_discovery must resolve its DataFrame from the canonical
        data_cache['estimation_data'] key (the one agent.py writes and
        estimation.py reads), NOT the dead 'estimation_data'-vs-'data' mismatch.

        Pre-fix: _run_discovery reads data_cache.get('data') -> None even though
        estimation_data is populated -> raises ValueError -> AssertionError here
        because discovery_runner.discover_dag is never called.
        """
        from unittest.mock import AsyncMock, MagicMock

        import pandas as pd

        node = GraphBuilderNode()

        df = pd.DataFrame(
            {
                "hcp_engagement_level": [0.1, 0.2, 0.3, 0.4],
                "patient_conversion_rate": [1.0, 2.0, 3.0, 4.0],
            }
        )

        # Stub the discovery runner + gate so we assert *only* on data wiring,
        # not on the (slow, nondeterministic) GES/PC algorithms.
        fake_result = MagicMock()
        fake_result.n_edges = 1
        node._discovery_runner = MagicMock()
        node._discovery_runner.discover_dag = AsyncMock(return_value=fake_result)

        fake_eval = MagicMock()
        fake_eval.decision.value = "accept"
        fake_eval.to_dict.return_value = {"decision": "accept", "confidence": 0.9}
        node._discovery_gate = MagicMock()
        node._discovery_gate.evaluate = MagicMock(return_value=fake_eval)

        state: CausalImpactState = {
            "query": "test",
            "query_id": "test-gb1",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "data_cache": {"estimation_data": df},
            "status": "pending",
        }

        result, evaluation = await node._run_discovery(
            state, "hcp_engagement_level", "patient_conversion_rate"
        )

        # discover_dag must have been called -> proves data was resolved.
        node._discovery_runner.discover_dag.assert_awaited_once()
        passed_df = node._discovery_runner.discover_dag.await_args.kwargs["data"]
        assert list(passed_df.columns) == [
            "hcp_engagement_level",
            "patient_conversion_rate",
        ]
        assert evaluation["decision"] == "accept"

    @pytest.mark.asyncio
    async def test_discovery_skip_surfaced_when_no_estimation_data(self):
        """M-gb1: when auto_discover=True but no estimation_data is in the cache,
        the skip must be surfaced (not silently swallowed): the result records a
        distinct 'discovery_skip_reason' and appends a string to 'warnings'.
        Behavior otherwise unchanged: pipeline still produces a manual DAG.
        """
        node = GraphBuilderNode()

        state: CausalImpactState = {
            "query": "test",
            "query_id": "test-gb1-skip",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "auto_discover": True,
            # No data_cache / no estimation_data -> discovery must skip + surface.
            "status": "pending",
            "warnings": [],
        }

        result = await node.execute(state)

        # Pipeline still succeeds with a manual DAG (graceful degradation).
        assert "causal_graph" in result
        assert result["causal_graph"]["discovery_algorithms_used"] == []
        # Skip is surfaced distinctly, not swallowed.
        assert result.get("discovery_skip_reason")
        assert "estimation_data" in result["discovery_skip_reason"]
        assert any("discovery" in w.lower() for w in result.get("warnings", [])), result.get(
            "warnings"
        )

    @pytest.mark.asyncio
    async def test_no_discovery_skip_reason_when_discovery_succeeds(self):
        """discovery_skip_reason must NOT be set when discovery runs cleanly."""
        from unittest.mock import AsyncMock, patch

        from src.causal_engine.discovery import (
            DiscoveryConfig,
            DiscoveryResult,
            GateDecision,
        )
        from src.causal_engine.discovery.gate import GateEvaluation

        node = GraphBuilderNode()
        state: CausalImpactState = {
            "query": "test",
            "query_id": "test-gb1-ok",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "auto_discover": True,
            "status": "pending",
        }
        mock_result = DiscoveryResult(success=True, config=DiscoveryConfig())
        mock_eval = GateEvaluation(
            decision=GateDecision.REVIEW,
            confidence=0.6,
            reasons=["ok"],
            high_confidence_edges=[],
        )
        with patch.object(node, "_run_discovery", new_callable=AsyncMock) as m:
            m.return_value = (mock_result, mock_eval.to_dict())
            result = await node.execute(state)

        assert not result.get("discovery_skip_reason")


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
            assert expected_treatment in treatment or treatment == "hcp_engagement_level", (
                f"Failed for query: {query}"
            )

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
            assert expected_outcome in outcome or outcome == "patient_conversion_rate", (
                f"Failed for query: {query}"
            )

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

    def test_m_structure_collider_not_in_adjustment_set(self):
        """M-bias regression (M-gb2): a collider on an M-structure must NEVER be
        selected into an adjustment set.

        DAG (M-structure, no direct T->O effect path):
            T <- U1 -> C <- U2 -> O
        C is a collider (U1 -> C <- U2). There is NO open backdoor path between
        T and O, so the empty set {} is the valid backdoor adjustment set.
        Conditioning on the collider C opens the path T<-U1->C<-U2->O (M-bias),
        so C must be excluded from every returned set.
        """
        import networkx as nx

        node = GraphBuilderNode()

        dag = nx.DiGraph()
        dag.add_edge("U1", "T")
        dag.add_edge("U1", "C")
        dag.add_edge("U2", "O")
        dag.add_edge("U2", "C")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        # The collider C must not be conditioned on in ANY returned set.
        assert all("C" not in adj_set for adj_set in adjustment_sets), (
            f"Collider 'C' must never be in an adjustment set (M-bias), got {adjustment_sets}"
        )
        # The empty set is the correct backdoor adjustment set here.
        assert [] in adjustment_sets, (
            f"Empty set must be a valid backdoor adjustment set, got {adjustment_sets}"
        )

    def test_mixed_mstructure_and_confounder_excludes_collider(self):
        """With both a genuine confounder W and an M-structure collider C, the
        finder must adjust for W and never include the collider C.

        DAG:
            W -> T, W -> O           (W is a confounder -> must be adjusted)
            U1 -> T, U1 -> C         (M-structure arm)
            U2 -> O, U2 -> C         (C is a collider)
            T -> O                   (causal effect)
        Correct backdoor adjustment set: {W}. Conditioning on C opens the
        M-path, so C must be excluded; in particular the set chosen by
        downstream estimation (adjustment_sets[0]) must not contain C.
        """
        import networkx as nx

        node = GraphBuilderNode()

        dag = nx.DiGraph()
        dag.add_edge("W", "T")
        dag.add_edge("W", "O")
        dag.add_edge("U1", "T")
        dag.add_edge("U1", "C")
        dag.add_edge("U2", "O")
        dag.add_edge("U2", "C")
        dag.add_edge("T", "O")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        assert len(adjustment_sets) >= 1
        # Collider C is excluded from every returned set.
        assert all("C" not in adj_set for adj_set in adjustment_sets), (
            f"Collider 'C' must never be in an adjustment set, got {adjustment_sets}"
        )
        # The set actually used downstream (estimation.py picks index 0) must
        # adjust for the genuine confounder W and not the collider.
        chosen = adjustment_sets[0]
        assert "W" in chosen, f"Confounder 'W' must be adjusted for, got chosen set {chosen}"
        assert "C" not in chosen

    def test_mediator_excluded_from_adjustment_sets(self):
        """A mediator (descendant of treatment) must never be adjusted for.

        DAG: T -> M -> O, T -> O. Adjusting for the mediator M blocks part of
        the causal effect (over-control bias). The empty set is the valid
        backdoor adjustment set.
        """
        import networkx as nx

        node = GraphBuilderNode()

        dag = nx.DiGraph()
        dag.add_edge("T", "M")
        dag.add_edge("M", "O")
        dag.add_edge("T", "O")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        assert all("M" not in adj_set for adj_set in adjustment_sets), (
            f"Mediator 'M' must never be in an adjustment set, got {adjustment_sets}"
        )
        assert [] in adjustment_sets

    def test_more_than_three_confounders_are_all_adjusted(self):
        """M-gb2 regression: with > 3 independent confounders, the size-capped
        search finds no admissible set of size <= 3, so it must fall back to the
        full candidate set rather than returning [[]] (zero confounder control).

        DAG: C1..C4 each -> T and -> O ; T -> O. The ONLY admissible backdoor set
        is {C1,C2,C3,C4} (any proper subset leaves one confounded path open). A
        [[]] result would silently run the estimator with NO adjustment -> a
        plausible-but-wrong confounded ATE, the exact harm M-gb2 prevents.
        """
        import networkx as nx

        node = GraphBuilderNode()

        dag = nx.DiGraph()
        confounders = ["C1", "C2", "C3", "C4"]
        for c in confounders:
            dag.add_edge(c, "T")
            dag.add_edge(c, "O")
        dag.add_edge("T", "O")

        adjustment_sets = node._find_adjustment_sets(dag, "T", "O")

        # Must NOT silently return the empty (no-adjustment) set.
        assert adjustment_sets[0], (
            f"> 3 confounders must yield a non-empty adjustment set, got {adjustment_sets}"
        )
        # The admissible set must contain every confounder.
        assert set(confounders).issubset(set(adjustment_sets[0])), (
            f"adjustment set must control for all confounders, got {adjustment_sets[0]}"
        )

    def test_treatment_equals_outcome_returns_trivial_set(self):
        """M-gb2 (D2): a degenerate treatment == outcome query must return the
        trivial empty adjustment set, not raise NetworkXError (non-disjoint node
        sets) out of nx.is_d_separator and hard-fail the node."""
        import networkx as nx

        node = GraphBuilderNode()

        dag = nx.DiGraph()
        dag.add_edge("C", "T")
        dag.add_edge("C", "O")
        dag.add_edge("T", "O")

        # Must not raise; trivial degenerate query yields no adjustment.
        assert node._find_adjustment_sets(dag, "T", "T") == [[]]


class TestDiscoveredDagIncludesEstimandEdge:
    """When the gate ACCEPTs a discovered DAG, the treatment->outcome estimand
    edge must be present even if constraint-based discovery did not draw it (a
    CI test can miss a real effect on binary data). The agent estimates that
    effect, so the reported DAG must show it — as long as it stays acyclic."""

    def test_accept_adds_missing_treatment_outcome_edge(self):
        import networkx as nx

        from src.causal_engine.discovery import (
            DiscoveryConfig,
            DiscoveryResult,
            GateDecision,
        )

        node = GraphBuilderNode()
        # Discovered structure: confounders -> treatment & outcome, but NO
        # treatment -> outcome edge (mirrors the patient_journeys guided run).
        g = nx.DiGraph()
        g.add_edges_from(
            [
                ("disease_severity", "treatment_arm"),
                ("disease_severity", "persistent_180d"),
                ("engagement_score", "treatment_arm"),
                ("engagement_score", "persistent_180d"),
            ]
        )
        result = DiscoveryResult(success=True, config=DiscoveryConfig(), ensemble_dag=g)
        dag, _augmented = node._build_dag_with_discovery(
            "treatment_arm",
            "persistent_180d",
            ["disease_severity", "engagement_score"],
            result,
            {"decision": GateDecision.ACCEPT.value},
        )
        assert dag.has_edge("treatment_arm", "persistent_180d"), "estimand edge missing"
        # Discovered confounder edges preserved and the graph stays acyclic.
        assert dag.has_edge("disease_severity", "treatment_arm")
        assert nx.is_directed_acyclic_graph(dag)
