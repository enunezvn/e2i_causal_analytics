"""Unit tests for causal-chain filtering in the graph route (ai-insights viz).

The knowledge-graph walk can emit non-simple paths that revisit a node (rendered
as nonsensical loops in the Active Causal Chains viz) and several near-duplicate
paths differing only in a confidence value. ``_clean_causal_chains`` must drop the
degenerate ones and de-duplicate so the viz shows sensible, distinct chains.
"""

from __future__ import annotations

from src.api.models.graph import GraphNode, GraphPath, GraphRelationship
from src.api.routes.graph import _clean_causal_chains, _is_simple_chain


def _path(node_ids: list[str], conf: float) -> GraphPath:
    nodes = [GraphNode(id=nid, type="Variable", name=nid) for nid in node_ids]
    rels = [
        GraphRelationship(
            id=f"r{i}",
            type="CAUSES",
            source_id=node_ids[i],
            target_id=node_ids[i + 1],
            confidence=conf,
        )
        for i in range(len(node_ids) - 1)
    ]
    return GraphPath(nodes=nodes, relationships=rels, total_confidence=conf, path_length=len(rels))


def test_is_simple_chain_rejects_repeated_node():
    # A -> B -> A is a cycle that renders as a nonsensical loop.
    assert _is_simple_chain(_path(["a", "b", "a"], 0.8)) is False


def test_is_simple_chain_rejects_single_node_or_no_rel():
    assert _is_simple_chain(_path(["a"], 0.8)) is False
    lone = GraphPath(
        nodes=[
            GraphNode(id="a", type="Variable", name="a"),
            GraphNode(id="b", type="Variable", name="b"),
        ],
        relationships=[],
        total_confidence=0.8,
        path_length=0,
    )
    assert _is_simple_chain(lone) is False


def test_is_simple_chain_accepts_simple_path():
    assert _is_simple_chain(_path(["a", "b", "c"], 0.8)) is True


def test_clean_drops_cyclic_chains():
    cleaned = _clean_causal_chains([_path(["a", "b", "c"], 0.9), _path(["x", "y", "x"], 0.7)])
    assert len(cleaned) == 1
    assert [n.id for n in cleaned[0].nodes] == ["a", "b", "c"]


def test_clean_dedupes_keeping_highest_confidence():
    cleaned = _clean_causal_chains(
        [_path(["a", "b"], 0.5), _path(["a", "b"], 0.9), _path(["a", "b"], 0.3)]
    )
    assert len(cleaned) == 1
    assert cleaned[0].total_confidence == 0.9


def test_clean_preserves_first_seen_order():
    cleaned = _clean_causal_chains(
        [_path(["a", "b"], 0.8), _path(["c", "d"], 0.8), _path(["a", "b"], 0.9)]
    )
    assert [[n.id for n in c.nodes] for c in cleaned] == [["a", "b"], ["c", "d"]]


def test_clean_no_surviving_chain_repeats_a_node():
    cleaned = _clean_causal_chains([_path(["a", "b", "c", "b"], 0.8), _path(["m", "n"], 0.8)])
    for c in cleaned:
        ids = [n.id for n in c.nodes]
        assert len(set(ids)) == len(ids)
