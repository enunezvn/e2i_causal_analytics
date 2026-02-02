"""Tests for DiscoveryGate class.

Version: 1.0.0
Tests the gating logic for causal discovery results.
"""

import networkx as nx
import numpy as np
import pytest

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    GateDecision,
)
from src.causal_engine.discovery.gate import (
    DiscoveryGate,
    GateConfig,
    GateEvaluation,
)


class TestGateConfig:
    """Test GateConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GateConfig()

        assert config.accept_threshold == 0.8
        assert config.review_threshold == 0.5
        assert config.augment_edge_threshold == 0.9
        assert config.min_algorithm_agreement == 0.5
        assert config.max_rejected_edges_fraction == 0.3
        assert config.min_edges == 1
        assert config.require_dag is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GateConfig(
            accept_threshold=0.9,
            review_threshold=0.6,
            min_edges=3,
        )

        assert config.accept_threshold == 0.9
        assert config.review_threshold == 0.6
        assert config.min_edges == 3


class TestGateEvaluation:
    """Test GateEvaluation dataclass."""

    def test_create_evaluation(self):
        """Test creating a GateEvaluation."""
        edge = DiscoveredEdge(source="A", target="B", confidence=0.95)

        evaluation = GateEvaluation(
            decision=GateDecision.ACCEPT,
            confidence=0.85,
            reasons=["High confidence"],
            high_confidence_edges=[edge],
            warnings=[],
        )

        assert evaluation.decision == GateDecision.ACCEPT
        assert evaluation.confidence == 0.85
        assert len(evaluation.high_confidence_edges) == 1

    def test_evaluation_to_dict(self):
        """Test GateEvaluation.to_dict() serialization."""
        evaluation = GateEvaluation(
            decision=GateDecision.REVIEW,
            confidence=0.65,
            reasons=["Medium confidence", "Expert review recommended"],
            high_confidence_edges=[],
            rejected_edges=[DiscoveredEdge(source="X", target="Y", confidence=0.3)],
            warnings=["Some edges have low confidence"],
        )

        d = evaluation.to_dict()

        assert d["decision"] == "review"
        assert d["confidence"] == 0.65
        assert len(d["reasons"]) == 2
        assert d["n_high_confidence_edges"] == 0
        assert d["n_rejected_edges"] == 1


class TestDiscoveryGate:
    """Test DiscoveryGate class."""

    @pytest.fixture
    def gate(self):
        """Create a DiscoveryGate with default config."""
        return DiscoveryGate()

    @pytest.fixture
    def high_confidence_result(self):
        """Create a high-confidence discovery result."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.95, algorithm_votes=3),
            DiscoveredEdge(source="B", target="C", confidence=0.90, algorithm_votes=3),
            DiscoveredEdge(source="A", target="C", confidence=0.85, algorithm_votes=2),
        ]

        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C"), ("A", "C")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C"), ("A", "C")],
                runtime_seconds=1.5,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.LINGAM,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C")],
                runtime_seconds=2.0,
                converged=True,
            ),
        ]

        return DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=algorithm_results,
        )

    @pytest.fixture
    def low_confidence_result(self):
        """Create a low-confidence discovery result."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("C", "D")])

        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.4, algorithm_votes=1),
            DiscoveredEdge(source="C", target="D", confidence=0.35, algorithm_votes=1),
        ]

        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((4, 4)),
                edge_list=[("A", "B")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((4, 4)),
                edge_list=[("C", "D")],
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        return DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=algorithm_results,
        )

    def test_evaluate_high_confidence_accepts(self, gate, high_confidence_result):
        """Test that high confidence results are accepted."""
        evaluation = gate.evaluate(high_confidence_result)

        assert evaluation.decision == GateDecision.ACCEPT
        assert evaluation.confidence >= 0.8
        assert len(evaluation.reasons) > 0

    def test_evaluate_low_confidence_rejects(self, gate, low_confidence_result):
        """Test that low confidence results are rejected."""
        evaluation = gate.evaluate(low_confidence_result)

        # Low confidence should result in REJECT or AUGMENT (if high-conf edges exist)
        assert evaluation.decision in [GateDecision.REJECT, GateDecision.REVIEW]
        assert evaluation.confidence < 0.8

    def test_evaluate_failed_discovery(self, gate):
        """Test evaluation of failed discovery."""
        config = DiscoveryConfig()
        result = DiscoveryResult(
            success=False,
            config=config,
            metadata={"error": "Algorithm failed to converge"},
        )

        evaluation = gate.evaluate(result)

        assert evaluation.decision == GateDecision.REJECT
        assert evaluation.confidence == 0.0
        assert "Discovery failed" in evaluation.reasons

    def test_evaluate_no_edges(self, gate):
        """Test evaluation when no edges discovered."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_nodes_from(["A", "B", "C"])  # Nodes but no edges

        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=[],
        )

        evaluation = gate.evaluate(result)

        assert evaluation.decision == GateDecision.REJECT
        assert "Too few edges" in evaluation.reasons[0]

    def test_evaluate_with_expected_edges(self, gate, high_confidence_result):
        """Test evaluation with expected edges for validation."""
        expected_edges = [("A", "B"), ("B", "C"), ("X", "Y")]  # One missing

        evaluation = gate.evaluate(high_confidence_result, expected_edges=expected_edges)

        # Should include recall/precision in reasons
        assert any("recall" in r.lower() for r in evaluation.reasons)

    def test_high_confidence_edges_populated(self, gate, high_confidence_result):
        """Test that high confidence edges are properly populated."""
        evaluation = gate.evaluate(high_confidence_result)

        # Edges with confidence >= 0.9 (augment_edge_threshold) should be in high_confidence_edges
        assert len(evaluation.high_confidence_edges) >= 1
        assert all(e.confidence >= 0.9 for e in evaluation.high_confidence_edges)

    def test_custom_config_thresholds(self):
        """Test gate with custom thresholds."""
        strict_config = GateConfig(
            accept_threshold=0.95,
            review_threshold=0.8,
        )
        gate = DiscoveryGate(config=strict_config)

        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edge("A", "B")

        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.85, algorithm_votes=2),
        ]

        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((2, 2)),
                edge_list=[("A", "B")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((2, 2)),
                edge_list=[("A", "B")],
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=algorithm_results,
        )

        evaluation = gate.evaluate(result)

        # With strict thresholds, 0.85 confidence should not be ACCEPT
        assert evaluation.decision != GateDecision.ACCEPT

    def test_should_accept(self, gate, high_confidence_result):
        """Test should_accept convenience method."""
        assert gate.should_accept(high_confidence_result) is True

    def test_should_not_accept_failed(self, gate):
        """Test should_accept returns False for failed discovery."""
        config = DiscoveryConfig()
        result = DiscoveryResult(success=False, config=config)

        assert gate.should_accept(result) is False

    def test_get_augmentation_edges(self, gate):
        """Test getting augmentation edges."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])

        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.95, algorithm_votes=2),
            DiscoveredEdge(source="B", target="C", confidence=0.92, algorithm_votes=2),
            DiscoveredEdge(source="A", target="C", confidence=0.5, algorithm_votes=1),  # Low conf
        ]

        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C")],
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=algorithm_results,
        )

        # Manual DAG with only one edge
        manual_dag = nx.DiGraph()
        manual_dag.add_edge("A", "B")

        augment_edges = gate.get_augmentation_edges(result, manual_dag)

        # Should return B->C (high confidence, not in manual DAG)
        assert len(augment_edges) >= 1
        edge_tuples = [(e.source, e.target) for e in augment_edges]
        assert ("B", "C") in edge_tuples

    def test_augmentation_avoids_cycles(self, gate):
        """Test that augmentation edges don't create cycles."""
        config = DiscoveryConfig()
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])  # Would be cyclic

        edges = [
            DiscoveredEdge(source="A", target="B", confidence=0.95, algorithm_votes=2),
            DiscoveredEdge(source="B", target="C", confidence=0.95, algorithm_votes=2),
            DiscoveredEdge(
                source="C", target="A", confidence=0.95, algorithm_votes=2
            ),  # Creates cycle
        ]

        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C"), ("C", "A")],
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=algorithm_results,
        )

        # Manual DAG: A -> B -> C (already has chain)
        manual_dag = nx.DiGraph()
        manual_dag.add_edges_from([("A", "B"), ("B", "C")])

        augment_edges = gate.get_augmentation_edges(result, manual_dag)

        # Should not include C -> A as it would create cycle
        edge_tuples = [(e.source, e.target) for e in augment_edges]
        assert ("C", "A") not in edge_tuples

    def test_disconnected_graph_penalty(self, gate):
        """Test that disconnected graphs receive lower structure score."""
        config = DiscoveryConfig()

        # Connected DAG
        connected_dag = nx.DiGraph()
        connected_dag.add_edges_from([("A", "B"), ("B", "C")])

        # Disconnected DAG (two components)
        disconnected_dag = nx.DiGraph()
        disconnected_dag.add_edges_from([("A", "B"), ("C", "D")])

        edges_connected = [
            DiscoveredEdge(source="A", target="B", confidence=0.9, algorithm_votes=2),
            DiscoveredEdge(source="B", target="C", confidence=0.9, algorithm_votes=2),
        ]

        edges_disconnected = [
            DiscoveredEdge(source="A", target="B", confidence=0.9, algorithm_votes=2),
            DiscoveredEdge(source="C", target="D", confidence=0.9, algorithm_votes=2),
        ]

        algorithm_results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((4, 4)),
                edge_list=[],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((4, 4)),
                edge_list=[],
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        result_connected = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=connected_dag,
            edges=edges_connected,
            algorithm_results=algorithm_results,
        )

        result_disconnected = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=disconnected_dag,
            edges=edges_disconnected,
            algorithm_results=algorithm_results,
        )

        eval_connected = gate.evaluate(result_connected)
        eval_disconnected = gate.evaluate(result_disconnected)

        # Structure score for connected should be higher
        assert (
            eval_connected.metadata["structure_score"]
            >= eval_disconnected.metadata["structure_score"]
        )
