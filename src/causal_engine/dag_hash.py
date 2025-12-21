"""
DAG Version Hashing Utility.

Provides deterministic SHA256 hashing of causal DAG structures for:
- Version tracking of DAG modifications
- Expert approval linkage
- Change detection in quarterly audits

Version: 4.3
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import importlib.util

NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None


def compute_dag_hash(
    causal_graph: Optional[Dict[str, Any]] = None,
    dag: Optional[Any] = None,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
) -> str:
    """
    Compute deterministic SHA256 hash of a causal DAG structure.

    The hash is based on:
    - Sorted node list
    - Sorted edge list
    - Treatment and outcome variables (if provided)

    This ensures the same DAG always produces the same hash,
    regardless of how it was constructed.

    Args:
        causal_graph: CausalGraph dict with 'nodes', 'edges', 'treatment_nodes', 'outcome_nodes'
        dag: NetworkX DiGraph (alternative to causal_graph)
        treatment: Treatment variable name (if not in causal_graph)
        outcome: Outcome variable name (if not in causal_graph)

    Returns:
        64-character SHA256 hex digest

    Examples:
        >>> graph = {"nodes": ["A", "B", "C"], "edges": [("A", "B"), ("B", "C")]}
        >>> hash1 = compute_dag_hash(causal_graph=graph)
        >>> # Same structure produces same hash
        >>> graph2 = {"nodes": ["C", "A", "B"], "edges": [("B", "C"), ("A", "B")]}
        >>> hash2 = compute_dag_hash(causal_graph=graph2)
        >>> assert hash1 == hash2
    """
    # Extract components based on input type
    if causal_graph is not None:
        nodes = causal_graph.get("nodes", [])
        edges = causal_graph.get("edges", [])
        treatment_nodes = causal_graph.get("treatment_nodes", [])
        outcome_nodes = causal_graph.get("outcome_nodes", [])

        # Use provided treatment/outcome if not in graph
        if not treatment_nodes and treatment:
            treatment_nodes = [treatment]
        if not outcome_nodes and outcome:
            outcome_nodes = [outcome]

    elif dag is not None and NETWORKX_AVAILABLE:
        nodes = list(dag.nodes())
        edges = list(dag.edges())
        treatment_nodes = [treatment] if treatment else []
        outcome_nodes = [outcome] if outcome else []

    else:
        raise ValueError("Must provide either causal_graph dict or NetworkX dag")

    # Create canonical representation
    canonical = _create_canonical_representation(
        nodes=nodes,
        edges=edges,
        treatment_nodes=treatment_nodes,
        outcome_nodes=outcome_nodes,
    )

    # Compute SHA256 hash
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _create_canonical_representation(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    treatment_nodes: List[str],
    outcome_nodes: List[str],
) -> str:
    """
    Create canonical JSON representation for hashing.

    Sorting ensures deterministic output regardless of input order.

    Args:
        nodes: List of node names
        edges: List of (source, target) tuples
        treatment_nodes: List of treatment variable names
        outcome_nodes: List of outcome variable names

    Returns:
        Canonical JSON string
    """
    # Sort all lists for deterministic ordering
    sorted_nodes = sorted(nodes)
    sorted_edges = sorted([tuple(e) for e in edges])  # Ensure tuples
    sorted_treatment = sorted(treatment_nodes)
    sorted_outcome = sorted(outcome_nodes)

    # Create canonical structure
    canonical_dict = {
        "nodes": sorted_nodes,
        "edges": [[e[0], e[1]] for e in sorted_edges],
        "treatment_nodes": sorted_treatment,
        "outcome_nodes": sorted_outcome,
    }

    # Serialize with sorted keys for consistency
    return json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))


def compute_dag_hash_from_dot(dot_string: str) -> str:
    """
    Compute DAG hash from DOT format string.

    Parses the DOT string to extract nodes and edges, then computes hash.

    Args:
        dot_string: DOT format DAG representation

    Returns:
        64-character SHA256 hex digest
    """
    nodes = []
    edges = []

    for line in dot_string.split("\n"):
        line = line.strip()

        # Parse node definitions: "node_name" [label="..."];
        if line.startswith('"') and "[label=" in line:
            node = line.split('"')[1]
            nodes.append(node)

        # Parse edge definitions: "source" -> "target";
        elif "->" in line:
            parts = line.split("->")
            if len(parts) == 2:
                source = parts[0].strip().strip('"')
                target = parts[1].strip().rstrip(";").strip().strip('"')
                edges.append((source, target))

    return compute_dag_hash(causal_graph={"nodes": nodes, "edges": edges})


def is_dag_changed(
    old_hash: str,
    new_graph: Optional[Dict[str, Any]] = None,
    new_dag: Optional[Any] = None,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
) -> bool:
    """
    Check if a DAG has changed from a previous version.

    Args:
        old_hash: Previous DAG hash to compare against
        new_graph: Current CausalGraph dict
        new_dag: Current NetworkX DiGraph
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        True if DAG has changed, False if same
    """
    new_hash = compute_dag_hash(
        causal_graph=new_graph,
        dag=new_dag,
        treatment=treatment,
        outcome=outcome,
    )
    return old_hash != new_hash


def get_dag_changes(
    old_graph: Dict[str, Any],
    new_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two DAGs and identify changes.

    Args:
        old_graph: Previous CausalGraph
        new_graph: Current CausalGraph

    Returns:
        Dict with added/removed nodes and edges
    """
    old_nodes = set(old_graph.get("nodes", []))
    new_nodes = set(new_graph.get("nodes", []))

    old_edges = {tuple(e) for e in old_graph.get("edges", [])}
    new_edges = {tuple(e) for e in new_graph.get("edges", [])}

    return {
        "nodes_added": list(new_nodes - old_nodes),
        "nodes_removed": list(old_nodes - new_nodes),
        "edges_added": [list(e) for e in new_edges - old_edges],
        "edges_removed": [list(e) for e in old_edges - new_edges],
        "is_changed": old_nodes != new_nodes or old_edges != new_edges,
        "old_hash": compute_dag_hash(causal_graph=old_graph),
        "new_hash": compute_dag_hash(causal_graph=new_graph),
    }


def validate_dag_hash(hash_string: str) -> bool:
    """
    Validate that a string is a valid DAG hash format.

    Args:
        hash_string: Candidate hash string

    Returns:
        True if valid SHA256 hex digest, False otherwise
    """
    if not isinstance(hash_string, str):
        return False
    if len(hash_string) != 64:
        return False
    try:
        int(hash_string, 16)
        return True
    except ValueError:
        return False
