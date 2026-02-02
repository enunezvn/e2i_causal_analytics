"""Tests for causal discovery base classes.

Version: 1.0.0
Tests the base types, enums, and dataclasses for causal discovery.
"""

from uuid import uuid4

import networkx as nx
import numpy as np

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
    GateDecision,
)


class TestEnums:
    """Test ENUM types."""

    def test_discovery_algorithm_type_values(self):
        """Test DiscoveryAlgorithmType enum has correct values."""
        assert DiscoveryAlgorithmType.GES.value == "ges"
        assert DiscoveryAlgorithmType.PC.value == "pc"
        assert DiscoveryAlgorithmType.FCI.value == "fci"
        assert DiscoveryAlgorithmType.LINGAM.value == "lingam"
        assert DiscoveryAlgorithmType.DIRECT_LINGAM.value == "direct_lingam"
        assert DiscoveryAlgorithmType.ICA_LINGAM.value == "ica_lingam"

    def test_gate_decision_values(self):
        """Test GateDecision enum has correct values."""
        assert GateDecision.ACCEPT.value == "accept"
        assert GateDecision.REVIEW.value == "review"
        assert GateDecision.REJECT.value == "reject"
        assert GateDecision.AUGMENT.value == "augment"

    def test_edge_type_values(self):
        """Test EdgeType enum has correct values."""
        assert EdgeType.DIRECTED.value == "directed"
        assert EdgeType.UNDIRECTED.value == "undirected"
        assert EdgeType.BIDIRECTED.value == "bidirected"


class TestDiscoveredEdge:
    """Test DiscoveredEdge dataclass."""

    def test_create_discovered_edge(self):
        """Test creating a DiscoveredEdge."""
        edge = DiscoveredEdge(
            source="A",
            target="B",
            edge_type=EdgeType.DIRECTED,
            confidence=0.95,
            algorithm_votes=3,
            algorithms=["ges", "pc", "lingam"],
        )

        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.edge_type == EdgeType.DIRECTED
        assert edge.confidence == 0.95
        assert edge.algorithm_votes == 3
        assert edge.algorithms == ["ges", "pc", "lingam"]

    def test_discovered_edge_defaults(self):
        """Test default values for DiscoveredEdge."""
        edge = DiscoveredEdge(source="X", target="Y")

        assert edge.edge_type == EdgeType.DIRECTED
        assert edge.confidence == 1.0
        assert edge.algorithm_votes == 1
        assert edge.algorithms == []

    def test_discovered_edge_to_dict(self):
        """Test DiscoveredEdge.to_dict() serialization."""
        edge = DiscoveredEdge(
            source="A",
            target="B",
            edge_type=EdgeType.BIDIRECTED,
            confidence=0.7,
            algorithm_votes=2,
            algorithms=["ges", "pc"],
        )

        d = edge.to_dict()

        assert d["source"] == "A"
        assert d["target"] == "B"
        assert d["edge_type"] == "bidirected"
        assert d["confidence"] == 0.7
        assert d["algorithm_votes"] == 2
        assert d["algorithms"] == ["ges", "pc"]


class TestDiscoveryConfig:
    """Test DiscoveryConfig dataclass."""

    def test_create_config_defaults(self):
        """Test creating DiscoveryConfig with defaults."""
        config = DiscoveryConfig()

        assert config.algorithms == [DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC]
        assert config.alpha == 0.05
        assert config.ensemble_threshold == 0.5
        assert config.max_iter == 10000
        assert config.random_state == 42
        assert config.score_func == "local_score_BIC"
        assert config.assume_linear is True
        assert config.assume_gaussian is False

    def test_create_config_custom(self):
        """Test creating DiscoveryConfig with custom values."""
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.FCI],
            alpha=0.01,
            max_cond_vars=3,
            ensemble_threshold=0.7,
            max_iter=5000,
            random_state=123,
        )

        assert config.algorithms == [DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.FCI]
        assert config.alpha == 0.01
        assert config.max_cond_vars == 3
        assert config.ensemble_threshold == 0.7
        assert config.max_iter == 5000
        assert config.random_state == 123

    def test_config_to_dict(self):
        """Test DiscoveryConfig.to_dict() serialization."""
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES],
            alpha=0.1,
        )

        d = config.to_dict()

        assert d["algorithms"] == ["ges"]
        assert d["alpha"] == 0.1
        assert d["ensemble_threshold"] == 0.5


class TestAlgorithmResult:
    """Test AlgorithmResult dataclass."""

    def test_create_algorithm_result(self):
        """Test creating an AlgorithmResult."""
        adj = np.array([[0, 1], [0, 0]])
        edges = [("A", "B")]

        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.GES,
            adjacency_matrix=adj,
            edge_list=edges,
            runtime_seconds=1.5,
            converged=True,
            score=-100.5,
        )

        assert result.algorithm == DiscoveryAlgorithmType.GES
        np.testing.assert_array_equal(result.adjacency_matrix, adj)
        assert result.edge_list == edges
        assert result.runtime_seconds == 1.5
        assert result.converged is True
        assert result.score == -100.5

    def test_algorithm_result_defaults(self):
        """Test default values for AlgorithmResult."""
        result = AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.zeros((2, 2)),
            edge_list=[],
            runtime_seconds=0.5,
        )

        assert result.converged is True
        assert result.score is None
        assert result.metadata == {}


class TestDiscoveryResult:
    """Test DiscoveryResult dataclass."""

    def test_create_discovery_result(self):
        """Test creating a DiscoveryResult."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])

        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.9, algorithm_votes=2),
            DiscoveredEdge(source="B", target="C", confidence=0.8, algorithm_votes=2),
        ]

        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            gate_decision=GateDecision.ACCEPT,
            gate_confidence=0.85,
        )

        assert result.success is True
        assert result.n_edges == 2
        assert result.n_nodes == 3
        assert result.gate_decision == GateDecision.ACCEPT
        assert result.gate_confidence == 0.85

    def test_discovery_result_defaults(self):
        """Test default values for DiscoveryResult."""
        config = DiscoveryConfig()
        result = DiscoveryResult(success=False, config=config)

        assert result.ensemble_dag is None
        assert result.edges == []
        assert result.algorithm_results == []
        assert result.gate_decision is None
        assert result.gate_confidence == 0.0
        assert result.n_edges == 0
        assert result.n_nodes == 0

    def test_get_high_confidence_edges(self):
        """Test getting high confidence edges."""
        config = DiscoveryConfig()
        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.95),
            DiscoveredEdge(source="B", target="C", confidence=0.6),
            DiscoveredEdge(source="C", target="D", confidence=0.85),
        ]

        result = DiscoveryResult(success=True, config=config, edges=edges)

        high_conf = result.get_high_confidence_edges(threshold=0.8)

        assert len(high_conf) == 2
        assert high_conf[0].source == "A"
        assert high_conf[1].source == "C"

    def test_algorithm_agreement(self):
        """Test algorithm agreement calculation."""
        config = DiscoveryConfig()
        edges = [
            DiscoveredEdge(source="A", target="B", algorithm_votes=3),
            DiscoveredEdge(source="B", target="C", algorithm_votes=2),
        ]
        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[],
                runtime_seconds=1.0,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[],
                runtime_seconds=1.0,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.LINGAM,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[],
                runtime_seconds=1.0,
            ),
        ]

        result = DiscoveryResult(
            success=True,
            config=config,
            edges=edges,
            algorithm_results=algorithm_results,
        )

        # Total votes = 3 + 2 = 5, max_votes = 3 algorithms * 2 edges = 6
        assert result.algorithm_agreement == 5 / 6

    def test_to_adjacency_matrix(self):
        """Test converting to adjacency matrix."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("A", "C"), ("B", "C")])

        edges = [
            DiscoveredEdge(source="A", target="B"),
            DiscoveredEdge(source="A", target="C"),
            DiscoveredEdge(source="B", target="C"),
        ]

        result = DiscoveryResult(success=True, config=config, ensemble_dag=dag, edges=edges)

        adj = result.to_adjacency_matrix(node_order=["A", "B", "C"])

        expected = np.array(
            [
                [0, 1, 1],  # A -> B, A -> C
                [0, 0, 1],  # B -> C
                [0, 0, 0],  # C has no outgoing
            ]
        )

        np.testing.assert_array_equal(adj, expected)

    def test_to_dict_serialization(self):
        """Test DiscoveryResult.to_dict() serialization."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edge("X", "Y")

        edges = [DiscoveredEdge(source="X", target="Y", confidence=0.9)]
        session_id = uuid4()

        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            gate_decision=GateDecision.REVIEW,
            gate_confidence=0.7,
            session_id=session_id,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["n_edges"] == 1
        assert d["n_nodes"] == 2
        assert d["gate_decision"] == "review"
        assert d["gate_confidence"] == 0.7
        assert d["session_id"] == str(session_id)
        assert len(d["edges"]) == 1
        assert d["edges"][0]["source"] == "X"

    def test_empty_result_to_dict(self):
        """Test serializing an empty/failed result."""
        config = DiscoveryConfig()
        result = DiscoveryResult(success=False, config=config)

        d = result.to_dict()

        assert d["success"] is False
        assert d["n_edges"] == 0
        assert d["gate_decision"] is None
        assert d["session_id"] is None
