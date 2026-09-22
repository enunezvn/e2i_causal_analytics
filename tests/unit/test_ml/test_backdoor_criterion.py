"""Lane B (real-data causal estimation, 2026-09-22): the backdoor criterion as a
light, shared function.

``GraphBuilderNode._satisfies_backdoor_criterion`` is the production adjustment
finder's criterion; the structural assembler needs the SAME criterion without
importing the causal agent (15 s import, measured). This file pins the
criterion's truths on hand-built graphs; the delegation from the node is pinned
in ``tests/unit/test_agents/test_causal_impact/test_graph_builder_backdoor_shared.py``.
"""

from __future__ import annotations

import networkx as nx

from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion


def _g(*edges):
    return nx.DiGraph(list(edges))


def test_confounder_blocks_and_empty_set_does_not():
    g = _g(("C", "T"), ("C", "Y"), ("T", "Y"))
    assert satisfies_backdoor_criterion(g, {"C"}, "T", "Y") is True
    assert satisfies_backdoor_criterion(g, set(), "T", "Y") is False


def test_descendant_of_treatment_is_never_admissible():
    # M is a mediator: conditioning on it violates rule (1).
    g = _g(("T", "M"), ("M", "Y"), ("T", "Y"))
    assert satisfies_backdoor_criterion(g, {"M"}, "T", "Y") is False
    assert satisfies_backdoor_criterion(g, set(), "T", "Y") is True


def test_m_structure_collider_opens_a_path():
    # T -> V <- U -> Y : V is T-downstream (rule 1) AND conditioning would open
    # T - V - U - Y; the empty set is admissible.
    g = _g(("T", "V"), ("U", "V"), ("U", "Y"), ("T", "Y"))
    assert satisfies_backdoor_criterion(g, {"V"}, "T", "Y") is False
    assert satisfies_backdoor_criterion(g, set(), "T", "Y") is True


def test_latent_confounder_is_unblockable_by_observed_child():
    # U -> T, U -> Y with U unobserved: nothing observed blocks it unless a
    # child of U that sits ON the path is conditioned. F <- U only (F off the
    # path) does not block; {U} itself does.
    g = _g(("U", "T"), ("U", "Y"), ("U", "F"), ("T", "Y"))
    assert satisfies_backdoor_criterion(g, {"F"}, "T", "Y") is False
    assert satisfies_backdoor_criterion(g, {"U"}, "T", "Y") is True


def test_treatment_or_outcome_in_set_is_rejected():
    g = _g(("C", "T"), ("C", "Y"), ("T", "Y"))
    assert satisfies_backdoor_criterion(g, {"C", "T"}, "T", "Y") is False
    assert satisfies_backdoor_criterion(g, {"C", "Y"}, "T", "Y") is False
