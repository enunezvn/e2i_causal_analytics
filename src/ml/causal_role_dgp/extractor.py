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

import networkx as nx


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
