"""Tests for DAG version hashing utility.

Version: 4.3
Tests compute_dag_hash and related functions.
"""

import pytest

from src.causal_engine import (
    compute_dag_hash,
    compute_dag_hash_from_dot,
    get_dag_changes,
    is_dag_changed,
    validate_dag_hash,
)


class TestComputeDagHash:
    """Test compute_dag_hash function."""

    def test_basic_hash_computation(self):
        """Test basic hash computation from causal graph dict."""
        graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
            "treatment_nodes": ["A"],
            "outcome_nodes": ["C"],
        }

        hash_result = compute_dag_hash(causal_graph=graph)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_deterministic_hashing(self):
        """Test that same structure produces same hash."""
        graph1 = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
            "treatment_nodes": ["A"],
            "outcome_nodes": ["C"],
        }

        graph2 = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
            "treatment_nodes": ["A"],
            "outcome_nodes": ["C"],
        }

        hash1 = compute_dag_hash(causal_graph=graph1)
        hash2 = compute_dag_hash(causal_graph=graph2)

        assert hash1 == hash2

    def test_order_independent_hashing(self):
        """Test that node/edge order doesn't affect hash."""
        graph1 = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
            "treatment_nodes": ["A"],
            "outcome_nodes": ["C"],
        }

        graph2 = {
            "nodes": ["C", "A", "B"],  # Different order
            "edges": [("B", "C"), ("A", "B")],  # Different order
            "treatment_nodes": ["A"],
            "outcome_nodes": ["C"],
        }

        hash1 = compute_dag_hash(causal_graph=graph1)
        hash2 = compute_dag_hash(causal_graph=graph2)

        assert hash1 == hash2

    def test_different_structures_different_hashes(self):
        """Test that different DAGs produce different hashes."""
        graph1 = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
        }

        graph2 = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "C"), ("B", "C")],  # Different edges
        }

        hash1 = compute_dag_hash(causal_graph=graph1)
        hash2 = compute_dag_hash(causal_graph=graph2)

        assert hash1 != hash2

    def test_additional_node_changes_hash(self):
        """Test that adding a node changes the hash."""
        graph1 = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }

        graph2 = {
            "nodes": ["A", "B", "C"],  # Added node
            "edges": [("A", "B")],
        }

        hash1 = compute_dag_hash(causal_graph=graph1)
        hash2 = compute_dag_hash(causal_graph=graph2)

        assert hash1 != hash2

    def test_treatment_outcome_in_hash(self):
        """Test that treatment/outcome are included in hash."""
        graph1 = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
            "treatment_nodes": ["A"],
            "outcome_nodes": ["B"],
        }

        graph2 = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
            "treatment_nodes": ["B"],  # Swapped
            "outcome_nodes": ["A"],
        }

        hash1 = compute_dag_hash(causal_graph=graph1)
        hash2 = compute_dag_hash(causal_graph=graph2)

        assert hash1 != hash2

    def test_with_treatment_outcome_params(self):
        """Test providing treatment/outcome as parameters."""
        graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }

        hash1 = compute_dag_hash(causal_graph=graph, treatment="A", outcome="B")
        hash2 = compute_dag_hash(
            causal_graph={
                **graph,
                "treatment_nodes": ["A"],
                "outcome_nodes": ["B"],
            }
        )

        assert hash1 == hash2

    def test_empty_graph(self):
        """Test hashing empty graph."""
        graph = {
            "nodes": [],
            "edges": [],
        }

        hash_result = compute_dag_hash(causal_graph=graph)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_missing_optional_fields(self):
        """Test hashing with missing optional fields."""
        graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
            # No treatment_nodes, outcome_nodes
        }

        hash_result = compute_dag_hash(causal_graph=graph)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_no_input_raises_error(self):
        """Test that no input raises ValueError."""
        with pytest.raises(ValueError):
            compute_dag_hash()


class TestComputeDagHashFromDot:
    """Test compute_dag_hash_from_dot function."""

    def test_parse_dot_format(self):
        """Test parsing DOT format string."""
        dot_string = """digraph CausalDAG {
  rankdir=LR;
  node [shape=box];

  "treatment" [label="Treatment"];
  "outcome" [label="Outcome"];

  "treatment" -> "outcome";
}"""

        hash_result = compute_dag_hash_from_dot(dot_string)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_dot_matches_dict_hash(self):
        """Test that DOT and dict representations produce same hash."""
        graph = {
            "nodes": ["treatment", "outcome"],
            "edges": [("treatment", "outcome")],
        }

        dot_string = """digraph CausalDAG {
  rankdir=LR;
  node [shape=box];

  "treatment" [label="Treatment"];
  "outcome" [label="Outcome"];

  "treatment" -> "outcome";
}"""

        dict_hash = compute_dag_hash(causal_graph=graph)
        dot_hash = compute_dag_hash_from_dot(dot_string)

        assert dict_hash == dot_hash


class TestIsDagChanged:
    """Test is_dag_changed function."""

    def test_same_dag_not_changed(self):
        """Test that same DAG is not marked as changed."""
        graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }

        old_hash = compute_dag_hash(causal_graph=graph)

        assert not is_dag_changed(old_hash, new_graph=graph)

    def test_different_dag_is_changed(self):
        """Test that different DAG is marked as changed."""
        old_graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }
        new_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
        }

        old_hash = compute_dag_hash(causal_graph=old_graph)

        assert is_dag_changed(old_hash, new_graph=new_graph)


class TestGetDagChanges:
    """Test get_dag_changes function."""

    def test_identify_added_nodes(self):
        """Test identification of added nodes."""
        old_graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }
        new_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B")],
        }

        changes = get_dag_changes(old_graph, new_graph)

        assert "C" in changes["nodes_added"]
        assert len(changes["nodes_removed"]) == 0

    def test_identify_removed_nodes(self):
        """Test identification of removed nodes."""
        old_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B")],
        }
        new_graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }

        changes = get_dag_changes(old_graph, new_graph)

        assert "C" in changes["nodes_removed"]
        assert len(changes["nodes_added"]) == 0

    def test_identify_added_edges(self):
        """Test identification of added edges."""
        old_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B")],
        }
        new_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
        }

        changes = get_dag_changes(old_graph, new_graph)

        assert ["B", "C"] in changes["edges_added"]
        assert len(changes["edges_removed"]) == 0

    def test_identify_removed_edges(self):
        """Test identification of removed edges."""
        old_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C")],
        }
        new_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B")],
        }

        changes = get_dag_changes(old_graph, new_graph)

        assert ["B", "C"] in changes["edges_removed"]
        assert len(changes["edges_added"]) == 0

    def test_is_changed_flag(self):
        """Test is_changed flag accuracy."""
        graph1 = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }
        graph2 = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }
        graph3 = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B")],
        }

        same_changes = get_dag_changes(graph1, graph2)
        diff_changes = get_dag_changes(graph1, graph3)

        assert not same_changes["is_changed"]
        assert diff_changes["is_changed"]

    def test_hash_values_included(self):
        """Test that old and new hashes are included."""
        old_graph = {
            "nodes": ["A", "B"],
            "edges": [("A", "B")],
        }
        new_graph = {
            "nodes": ["A", "B", "C"],
            "edges": [("A", "B")],
        }

        changes = get_dag_changes(old_graph, new_graph)

        assert "old_hash" in changes
        assert "new_hash" in changes
        assert len(changes["old_hash"]) == 64
        assert len(changes["new_hash"]) == 64
        assert changes["old_hash"] != changes["new_hash"]


class TestValidateDagHash:
    """Test validate_dag_hash function."""

    def test_valid_hash(self):
        """Test validation of valid hash."""
        valid_hash = "a" * 64

        assert validate_dag_hash(valid_hash)

    def test_actual_computed_hash(self):
        """Test validation of actually computed hash."""
        graph = {"nodes": ["A", "B"], "edges": [("A", "B")]}
        computed_hash = compute_dag_hash(causal_graph=graph)

        assert validate_dag_hash(computed_hash)

    def test_invalid_length(self):
        """Test rejection of wrong-length string."""
        short_hash = "a" * 32
        long_hash = "a" * 128

        assert not validate_dag_hash(short_hash)
        assert not validate_dag_hash(long_hash)

    def test_invalid_characters(self):
        """Test rejection of non-hex characters."""
        invalid_hash = "g" * 64  # 'g' is not hex

        assert not validate_dag_hash(invalid_hash)

    def test_non_string_input(self):
        """Test rejection of non-string input."""
        assert not validate_dag_hash(123)
        assert not validate_dag_hash(None)
        assert not validate_dag_hash(["a" * 64])

    def test_empty_string(self):
        """Test rejection of empty string."""
        assert not validate_dag_hash("")

    def test_uppercase_hex(self):
        """Test acceptance of uppercase hex."""
        uppercase_hash = "A" * 64

        assert validate_dag_hash(uppercase_hash)

    def test_mixed_case_hex(self):
        """Test acceptance of mixed case hex."""
        mixed_hash = "aAbBcCdDeEfF" + "0" * 52

        assert validate_dag_hash(mixed_hash)
