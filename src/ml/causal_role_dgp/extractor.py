"""Mechanical role extractor (plan §3.3).

Given a DAG ``G``, treatment node ``T``, and outcome node ``Y``, classify
any other node into one of the six Pearl-Lauritzen graph roles:
ancestor, confounder, mediator, collider, descendant, instrument.

Priority order is load-bearing (collider > confounder > mediator >
instrument > ancestor > descendant). The two-condition instrument check
(exclusion + exogeneity) is sufficient only for confounders represented
in G — for the synthetic DAGs we author this is always the case. See
plan §3.3 for the caveat that prevents overclaim against real-world IV
validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import networkx as nx

if TYPE_CHECKING:
    from src.data.feature_contract import FeatureContract


def _is_confounder_collider_m_structure(
    node: str,
    treatment: str,
    outcome: str,
    graph: nx.DiGraph,
) -> bool:
    """Detect the confounder-collider M-structure ``T → V ← U → Y`` (Issue #501).

    A node ``V`` is a confounder-collider M-structure w.r.t. ``(T, Y)`` iff ALL
    of (plan ``.claude/plans/501_ac35_gate_implementation_plan.md`` §1.2):

    - **(a)** ``V ∈ descendants(T)`` — T reaches V (the ``T → V`` arm).
    - **(b)** ``V ∉ descendants(Y)`` — V is NOT a literal common-descendant of Y
      (else the literal-collider rule, ``extract_role`` Step 1, already
      classified it; this guards against double-handling AND against the
      ``Y → V`` phantom-edge dodge).
    - **(c)** there exists a parent ``U ∈ predecessors(V)``, ``U ∉ {T, Y, V}``,
      with ``U ∉ descendants(T)`` (U is **independent** of T — not itself
      T-downstream).
    - **(d)** in the graph with ``V`` removed, there is a directed path
      ``U → … → Y`` (U reaches the outcome via a path that **bypasses V** — a
      genuine independent arrowhead into Y, opening the backdoor
      ``T → V ← U → Y``).

    The "remove V, then test U → Y" formulation (d) is what distinguishes the
    real M-structure from the ``U → V → Y`` case where U is merely an ancestor
    of a mediator (it would wrongly fire without the bypass test). It also makes
    the rule **non-fakeable** per anti-mocking: the only way to make it fire is
    to author a *real* independent parent U with a *real* ``U → Y`` path; a
    phantom ``Y → V`` edge is caught by condition (b), and asserting a ``U → Y``
    edge the derivation does not support is a false structural attestation, not a
    rule-game.

    This implements the discriminator the classifier already documents in prose
    at ``src/data/causal_role_classifier.py:282-298`` ("baseline severity is
    itself a T-Y confounder with an arrowhead into V") but ``extract_role`` never
    coded.

    This function is PURE: it does not mutate ``graph`` (condition (d) operates
    on a copy).
    """
    desc_t = nx.descendants(graph, treatment)
    # (a) T reaches V.
    if node not in desc_t:
        return False
    # (b) V is not a literal common-descendant of Y (Step 1 owns that shape).
    if node in nx.descendants(graph, outcome):
        return False
    # (c)+(d): find an independent second parent U that reaches Y bypassing V.
    for parent in graph.predecessors(node):
        if parent in (treatment, outcome, node):
            continue
        if parent in desc_t:
            # U must be independent of T — not itself T-downstream.
            continue
        graph_no_v = graph.copy()
        graph_no_v.remove_node(node)
        if (
            parent in graph_no_v
            and outcome in graph_no_v
            and nx.has_path(graph_no_v, parent, outcome)
        ):
            return True
    return False


def extract_role(
    node: str,
    treatment: str,
    outcome: str,
    graph: nx.DiGraph,
) -> str:
    """Classify ``node`` into one of six causal roles w.r.t. ``(T, Y)``.

    Args:
        node: The node to classify. Must not be ``T`` or ``Y`` itself.
        treatment: The treatment node ``T``.
        outcome: The outcome node ``Y``.
        graph: A DAG (``nx.DiGraph``). Cycles are not handled; callers
            are responsible for ensuring acyclicity.

    Returns:
        One of ``{"ancestor", "confounder", "mediator", "collider",
        "descendant", "instrument"}``.

    Raises:
        ValueError: If ``node`` cannot be classified under any of the
            six roles (i.e., it has no relation to either ``T`` or
            ``Y`` in the DAG). This signals a malformed scenario.
    """
    # Step 1: common descendant of T and Y → collider
    if (node in nx.descendants(graph, treatment)) and (node in nx.descendants(graph, outcome)):
        return "collider"

    # Step 1.5 (Issue #501): confounder-collider M-structure T → V ← U → Y
    # (independent second parent U into both V and Y) → collider. Placed
    # AFTER Step 1 (literal T → V ← Y is handled there; cond. (b) excludes it)
    # and BEFORE the mediator/descendant fork (Step 3), which would otherwise
    # mis-return ``descendant`` for these cases. The confounder check (Step 2)
    # is disjoint: a confounder is a *parent* of T, while the M-structure V is a
    # *descendant* of T, so ordering Step 1.5 before Step 2 is safe.
    if _is_confounder_collider_m_structure(node, treatment, outcome, graph):
        return "collider"

    # Step 2: direct parent of BOTH T and Y → confounder
    if graph.has_edge(node, treatment) and graph.has_edge(node, outcome):
        return "confounder"

    # Step 3: descendant of T → mediator (if path to Y) else descendant
    if node in nx.descendants(graph, treatment):
        if node != outcome and nx.has_path(graph, node, outcome):
            return "mediator"
        return "descendant"

    # Step 4: parent of T → instrument iff exclusion AND exogeneity
    if graph.has_edge(node, treatment):
        graph_no_t = graph.copy()
        graph_no_t.remove_node(treatment)
        directed_path_after_t_removed = outcome in graph_no_t and nx.has_path(
            graph_no_t, node, outcome
        )
        common_ancestors = nx.ancestors(graph, node) & nx.ancestors(graph, outcome)
        if (not directed_path_after_t_removed) and (not common_ancestors):
            return "instrument"
        # Falls through to ancestor if either IV condition fails.

    # Step 5: parent of Y on a non-T path → ancestor
    if nx.has_path(graph, node, outcome) and (node not in nx.descendants(graph, treatment)):
        return "ancestor"

    raise ValueError(
        f"unclassified node {node!r} under (T={treatment!r}, Y={outcome!r}); "
        f"no role applies — this signals a malformed scenario DAG."
    )


def derive_structural_role(
    contract: Optional[FeatureContract],
) -> tuple[Optional[str], Optional[str]]:
    """Derive the deterministic causal role from a feature's authored
    ``CausalStructureAttestation`` edges via :func:`extract_role`.

    Returns ``(role, error)``:

    * ``(role, None)`` — attestation present and classifiable;
    * ``(None, None)`` — contract is ``None`` or has no ``causal_structure``
      (un-attested — the common case, and the dark-launch default);
    * ``(None, "<message>")`` — ``extract_role`` raised (malformed/unclassifiable
      authored DAG); the caller routes the feature to review.

    Pure, deterministic, zero LLM cost. This is the single code path shared by the
    pre-LLM structural decider (the ``EnsembleVoter`` wiring) AND the post-LLM
    telemetry helper ``_apply_structural_attestation`` — graph-building lives here,
    not in two places.
    """
    if contract is None or contract.causal_structure is None:
        return None, None
    att = contract.causal_structure
    try:
        graph = nx.DiGraph(list(att.edges))
        role = extract_role(att.feature_node, att.treatment_node, att.outcome_node, graph)
        return role, None
    except Exception as exc:  # noqa: BLE001 — author DAG errors must never crash the node
        return None, str(exc)
