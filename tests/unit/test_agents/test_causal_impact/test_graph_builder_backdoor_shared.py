"""Lane B (2026-09-22): ``GraphBuilderNode._satisfies_backdoor_criterion``
DELEGATES to the shared ``src.ml.causal_role_dgp.backdoor`` function — one
criterion, two consumers (the agent's adjustment finder and the structural
assembler). Pinned by substituting the shared function and watching the node's
answer change; a copy of the body would keep answering on its own.
"""

from __future__ import annotations

import networkx as nx

from src.agents.causal_impact.nodes import graph_builder as gb_mod
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode


def test_node_criterion_is_the_shared_function(monkeypatch):
    node = GraphBuilderNode()
    g = nx.DiGraph([("C", "T"), ("C", "Y"), ("T", "Y")])
    assert node._satisfies_backdoor_criterion(g, {"C"}, "T", "Y") is True

    calls = []

    def _fake(dag, adjustment_set, treatment, outcome):
        calls.append((sorted(adjustment_set), treatment, outcome))
        return False

    monkeypatch.setattr(gb_mod, "satisfies_backdoor_criterion", _fake)
    assert node._satisfies_backdoor_criterion(g, {"C"}, "T", "Y") is False
    assert calls == [(["C"], "T", "Y")]
