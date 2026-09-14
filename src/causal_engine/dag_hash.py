"""
DAG Version Hashing Utility.

Provides deterministic SHA256 hashing of causal DAG structures for:
- Version tracking of DAG modifications
- Expert approval linkage
- Change detection in quarterly audits

Version: 4.3
"""

import hashlib
import importlib.util
import json
from typing import Any, Dict, List, Optional, Tuple

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


def compute_adjustment_set_hash(adjustment_sets: List[List[str]]) -> str:
    """Compute deterministic SHA256 hash of an ``adjustment_sets`` list.

    Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §2.2.1
    (codex-2 B1 fix).

    Used by the Phase 2 adjustment-set policy node to track mutations to
    ``CausalGraph.adjustment_sets`` *independently* of ``dag_version_hash``.
    The base DAG hash is keyed by ``src/repositories/expert_review.py``
    (15 lookup sites) and must NEVER be mutated post-graph-builder; this
    separate hash carries adjustment-set provenance for the policy audit
    trail.

    The canonical form sorts inner lists then sorts outer list by the
    sorted-tuple of each inner — deterministic regardless of input order.

    Args:
        adjustment_sets: List of adjustment sets (each is a list of
            variable names). Empty list is permitted.

    Returns:
        64-character SHA256 hex digest.

    Examples:
        >>> h1 = compute_adjustment_set_hash([["A", "B"], ["C"]])
        >>> h2 = compute_adjustment_set_hash([["C"], ["B", "A"]])
        >>> assert h1 == h2  # same set; order-invariant
        >>> h3 = compute_adjustment_set_hash([["A"], ["C"]])
        >>> assert h1 != h3
    """
    canonical = json.dumps(
        [sorted(s) for s in sorted(adjustment_sets, key=lambda x: tuple(sorted(x)))],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adjustment_hash_from_snapshot(snapshot: Any) -> Optional[str]:
    """The adjustment-set hash a stored DAG snapshot PROVES, or None.

    A version row's ``adjustment_set_hash`` is NULL on every row migration 141
    backfilled and on every row a run without structure in scope recorded.
    Where the row kept a ``dag_structure_json`` snapshot, though, the snapshot
    NAMES the covariate set, so the hash is derived from it with the same
    function the writer uses -- one definition shared by the gate (is this run's
    structure already the last recorded one?) and by the review detail / queue
    (which recorded version is this review CURRENTLY on?).

    Inside a DICT snapshot an ABSENT, NULL or ``[]`` ``adjustment_sets`` all
    mean the EMPTY set, ``sha256("[]")`` -- the same value the run-side
    computation produces for a graph that conditions on nothing.

    A NULL (or otherwise non-dict) snapshot is UNKNOWN, not empty: the row
    records no structure at all, so it asserts nothing about the covariates and
    must never be read as the canonical empty set. Callers that need to PROVE a
    row is compatible therefore get None here and must decline.

    This READS STORED DATA and therefore NEVER RAISES (codex round 5, finding
    2). Migration 141's ``ck_erv_snapshot_object`` constrains the OUTER JSON
    type only, so ``{"adjustment_sets": [null]}`` is a row the table can hold,
    and handing it to ``compute_adjustment_set_hash`` raised ``TypeError``. The
    pending queue derives hashes OUTSIDE its 503-guarded database read, so one
    such version row aborted the whole queue response. The shape is validated
    first -- ``adjustment_sets`` must be absent, NULL, or a list whose every
    item is a list of strings -- and anything else is unprovable, which is what
    an unknown covariate set already means. Not the empty set: that is a real
    fact a malformed row does not establish.

    ``compute_adjustment_set_hash`` itself is unchanged. It is the WRITER's
    function, given a real ``CausalGraph.adjustment_sets``, and must keep
    raising on a caller's bug rather than hashing nonsense. Only this reader of
    stored data forgives.

    Args:
        snapshot: a stored ``dag_structure_json`` value, dict or NULL

    Returns:
        A 64-character SHA256 hex digest, or None when the snapshot proves
        nothing.
    """
    if not isinstance(snapshot, dict):
        return None
    adjustment_sets = snapshot.get("adjustment_sets")
    if adjustment_sets is None:
        adjustment_sets = []
    if not isinstance(adjustment_sets, list):
        return None
    for candidate in adjustment_sets:
        if not isinstance(candidate, list) or not all(isinstance(v, str) for v in candidate):
            return None
    return compute_adjustment_set_hash([list(s) for s in adjustment_sets])


def effective_adjustment_hash(adjustment_set_hash: Optional[str], snapshot: Any) -> Optional[str]:
    """The covariate set a ``(adjustment_set_hash, snapshot)`` half NAMES, or None.

    The stored adjustment hash when the column carries one; otherwise whatever
    the snapshot beside it PROVES, via ``adjustment_hash_from_snapshot``; None
    when neither half knows. A NULL adjustment column is "not recorded", not
    "no covariates", so the snapshot is the evidence -- and a NULL (or
    malformed) snapshot beside it proves nothing at all.

    This is the ONE rule the expert-review gate and the API share: the gate's
    version-match consult and the API's review-detail / pending-queue version
    selection each ask the same question of a stored version row -- "which
    covariate set does this row actually name?" -- and must get the same
    answer.

    Args:
        adjustment_set_hash: the stored ``adjustment_set_hash`` column, or None
        snapshot: the ``dag_structure_json`` snapshot recorded beside it

    Returns:
        A 64-character SHA256 hex digest, or None when neither half proves a
        covariate set.
    """
    if adjustment_set_hash is not None:
        return adjustment_set_hash
    return adjustment_hash_from_snapshot(snapshot)


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


#: The four keys ``compute_dag_hash`` reads off a ``causal_graph`` dict, each
#: with ``.get(key, [])`` -- so an explicit NULL reaches ``sorted(None)``.
_HASH_INPUT_KEYS = ("nodes", "edges", "treatment_nodes", "outcome_nodes")


def _graph_for_hash(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the hash inputs a stored snapshot may carry as NULL to empty lists.

    Every other key is preserved, and a graph carrying no NULL is returned
    unchanged -- so no hash VALUE moves; the only input whose behaviour changes
    is one that used to raise ``TypeError``.
    """
    if all(graph.get(key) is not None for key in _HASH_INPUT_KEYS):
        return graph
    return {**graph, **{key: graph.get(key) or [] for key in _HASH_INPUT_KEYS}}


def get_dag_changes(
    old_graph: Dict[str, Any],
    new_graph: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two DAGs and identify changes.

    The ADJUSTMENT SETS are part of the comparison (#1991 debt 3, spec §7): an
    expert review approves an estimand's *structure*, and the covariate set the
    estimator conditions on is structure -- two graphs with identical nodes and
    edges but different adjustment sets are different versions of the review, so
    an adjustment-set-only move reports ``is_changed`` True. Each set is
    compared as an unordered collection (sorted into a tuple), because the order
    covariates are listed in is not part of the identity.

    Every returned list is SORTED. These lists are serialized into the
    expert-review detail response (``DagChanges``), and a raw set difference
    iterates in hash order -- the same pair of graphs would render its diff
    differently between two calls, and across processes with different
    ``PYTHONHASHSEED``. Callers may rely on membership and on the order.

    A missing key and an explicit ``None`` are both read as empty
    (``.get(k) or []``): migration 141 backfilled version rows whose
    ``dag_structure_json`` carries a NULL ``adjustment_sets``, and ``.get(k, [])``
    would hand ``None`` to the comprehension. The same coercion is applied to
    the two graphs handed to ``compute_dag_hash`` (``_graph_for_hash``), which
    reads its four inputs with ``.get(k, [])`` and raised ``TypeError`` on an
    explicit NULL -- a stored snapshot must never crash the diff. Only that
    previously-raising input changes; a graph without a NULL hashes as before.

    Args:
        old_graph: Previous CausalGraph
        new_graph: Current CausalGraph

    Returns:
        Dict with added/removed nodes, edges and adjustment sets, ``is_changed``,
        and both hashes
    """
    old_nodes = set(old_graph.get("nodes") or [])
    new_nodes = set(new_graph.get("nodes") or [])

    old_edges = {tuple(e) for e in old_graph.get("edges") or []}
    new_edges = {tuple(e) for e in new_graph.get("edges") or []}

    old_adj = {tuple(sorted(a)) for a in old_graph.get("adjustment_sets") or []}
    new_adj = {tuple(sorted(a)) for a in new_graph.get("adjustment_sets") or []}

    return {
        "nodes_added": sorted(new_nodes - old_nodes),
        "nodes_removed": sorted(old_nodes - new_nodes),
        "edges_added": [list(e) for e in sorted(new_edges - old_edges)],
        "edges_removed": [list(e) for e in sorted(old_edges - new_edges)],
        "adjustment_sets_added": [list(a) for a in sorted(new_adj - old_adj)],
        "adjustment_sets_removed": [list(a) for a in sorted(old_adj - new_adj)],
        "is_changed": (old_nodes != new_nodes or old_edges != new_edges or old_adj != new_adj),
        "old_hash": compute_dag_hash(causal_graph=_graph_for_hash(old_graph)),
        "new_hash": compute_dag_hash(causal_graph=_graph_for_hash(new_graph)),
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
