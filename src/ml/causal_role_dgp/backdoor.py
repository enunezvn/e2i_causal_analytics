"""Pearl's backdoor criterion as a light, shared function.

Lane B of the real-data causal estimation program (2026-09-22): the structural
assembler (``src.ml.causal_role_dgp.assembler``) needs the SAME admissibility
test the causal agent's adjustment finder uses
(``GraphBuilderNode._find_adjustment_sets`` → ``_satisfies_backdoor_criterion``),
without importing the agent package (measured 15 s import through
``src.agents.causal_impact``). The node now delegates to this function, so there
is one criterion, not two copies that can drift.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx


def satisfies_backdoor_criterion(
    dag: nx.DiGraph,
    adjustment_set: Iterable[str],
    treatment: str,
    outcome: str,
) -> bool:
    """True iff ``adjustment_set`` satisfies the backdoor criterion for (T, Y).

    Pearl (2009, Def. 3.3.1):
      1. no node in the set is a descendant of the treatment;
      2. the set d-separates treatment from outcome in the proper backdoor
         graph (the treatment's OUTGOING edges removed).

    This correctly EXCLUDES colliders (and their descendants): conditioning on a
    collider would open a non-causal path (M-bias). Treatment or outcome inside
    the set is rejected outright.
    """
    z = set(adjustment_set)
    if treatment in z or outcome in z:
        return False
    if treatment not in dag or outcome not in dag:
        return False
    # (1) No descendant of treatment may be in the adjustment set.
    if z & nx.descendants(dag, treatment):
        return False
    # (2) d-separation in the proper backdoor graph (remove T's out-edges).
    backdoor_graph = dag.copy()
    backdoor_graph.remove_edges_from(list(dag.out_edges(treatment)))
    return bool(nx.is_d_separator(backdoor_graph, {treatment}, {outcome}, z))
