"""Unit tests for the KG insight's page-parity scoping helpers.

These are the server-side ports of the /knowledge-graph page's client-side
pipeline (causalGoldStandardGraph / dedupeParallelEdges / variableNeighborhoodGraph
in frontend/src/pages/KnowledgeGraph.tsx). The assertions mirror the page's
semantics so the insight grounding and the rendered canvas cannot silently
diverge.

Shapes match SemanticMemory.list_nodes/list_relationships output: edge
properties (brand/region/confidence) are FLAT top-level keys, not nested.
"""

from src.insights.knowledge_graph import (
    causal_gold_standard_graph,
    dedupe_parallel_edges,
    variable_neighborhood,
)


def _node(nid: str, ntype: str = "Variable", name: str | None = None) -> dict:
    return {"id": nid, "name": name or nid.split(":", 1)[-1], "type": ntype}


def _edge(
    src: str,
    tgt: str,
    etype: str = "CAUSES",
    brand: str | None = None,
    region: str | None = None,
    confidence: float | None = 0.8,
    eid: str = "",
) -> dict:
    e: dict = {
        "id": eid or f"{src}-{etype}-{tgt}-{brand}-{region}",
        "source_id": src,
        "target_id": tgt,
        "type": etype,
        "confidence": confidence,
    }
    if brand is not None:
        e["brand"] = brand
    if region is not None:
        e["region"] = region
    return e


class TestDedupeParallelEdges:
    def test_collapses_brand_x_region_copies_keeping_highest_confidence(self):
        copies = [
            _edge("var:a", "var:b", brand=b, region=r, confidence=c)
            for (b, r, c) in [
                ("Kisqali", "west", 0.7),
                ("Kisqali", "south", 0.9),  # representative (highest confidence)
                ("Fabhalta", "west", 0.8),
            ]
        ]
        out = dedupe_parallel_edges(copies)
        assert len(out) == 1
        edge = out[0]
        assert edge["confidence"] == 0.9
        assert edge["brands"] == ["Fabhalta", "Kisqali"]
        assert edge["regions"] == ["south", "west"]
        assert edge["parallel_edge_count"] == 3
        # The per-instance brand/region are replaced by the merged aggregate.
        assert "brand" not in edge and "region" not in edge

    def test_direction_and_type_stay_distinct(self):
        edges = [
            _edge("var:a", "var:b", etype="CAUSES"),
            _edge("var:b", "var:a", etype="CAUSES"),
            _edge("var:a", "var:b", etype="INFLUENCES"),
        ]
        assert len(dedupe_parallel_edges(edges)) == 3

    def test_singleton_edge_gets_no_parallel_count(self):
        out = dedupe_parallel_edges([_edge("var:a", "var:b", brand="Kisqali")])
        assert len(out) == 1
        assert "parallel_edge_count" not in out[0]


class TestCausalGoldStandardGraph:
    NODES = [
        _node("var:treatment"),
        _node("var:outcome"),
        _node("var:adherence"),
        _node("var:persistence"),
        _node("kpi:trx", ntype="KPI"),
        _node("var:island"),  # touched by no edge -> dropped
    ]
    EDGES = [
        _edge("var:treatment", "var:outcome", brand="Kisqali"),
        # lowercase brand tag: the graph holds 'Kisqali' AND 'kisqali' dupes,
        # and the page matches case-insensitively.
        _edge("var:outcome", "kpi:trx", brand="kisqali"),
        _edge("var:adherence", "var:persistence", brand="Fabhalta"),
        # Untagged structural edge: kept under EVERY brand scope.
        _edge("kpi:trx", "var:persistence", etype="INFLUENCES", brand=None),
    ]

    def test_all_keeps_everything_but_isolated_nodes(self):
        nodes, rels = causal_gold_standard_graph(self.NODES, self.EDGES, "All")
        assert {n["id"] for n in nodes} == {
            "var:treatment",
            "var:outcome",
            "var:adherence",
            "var:persistence",
            "kpi:trx",
        }
        assert len(rels) == 4

    def test_brand_scope_keeps_tagged_case_insensitive_plus_structural(self):
        nodes, rels = causal_gold_standard_graph(self.NODES, self.EDGES, "Kisqali")
        ids = {n["id"] for n in nodes}
        # Fabhalta's chain is out; adherence drops entirely (only touched by the
        # Fabhalta edge); persistence survives via the untagged structural edge.
        assert "var:adherence" not in ids
        assert {"var:treatment", "var:outcome", "kpi:trx", "var:persistence"} <= ids
        types = sorted((r["source_id"], r["type"], r["target_id"]) for r in rels)
        assert ("var:adherence", "CAUSES", "var:persistence") not in types
        assert ("kpi:trx", "INFLUENCES", "var:persistence") in types

    def test_edges_with_missing_endpoint_nodes_are_dropped(self):
        # kpi:trx absent from the node list -> both its edges must not survive.
        nodes = [n for n in self.NODES if n["id"] != "kpi:trx"]
        _, rels = causal_gold_standard_graph(nodes, self.EDGES, "All")
        assert all("kpi:trx" not in (r["source_id"], r["target_id"]) for r in rels)


class TestVariableNeighborhood:
    # a -> b -> c chain, unrelated x -> y chain, structural KPI attached to b.
    NODES = [
        _node("var:a"),
        _node("var:b"),
        _node("var:c"),
        _node("var:x"),
        _node("var:y"),
        _node("kpi:k", ntype="KPI"),
    ]
    EDGES = [
        _edge("var:a", "var:b"),
        _edge("var:b", "var:c"),
        _edge("var:x", "var:y"),
        _edge("var:b", "kpi:k", etype="EXPLAINS"),
    ]

    def test_keeps_ancestors_descendants_and_structural_context(self):
        nodes, rels = variable_neighborhood(self.NODES, self.EDGES, "var:b")
        assert {n["id"] for n in nodes} == {"var:a", "var:b", "var:c", "kpi:k"}
        kinds = {(r["source_id"], r["target_id"]) for r in rels}
        assert kinds == {("var:a", "var:b"), ("var:b", "var:c"), ("var:b", "kpi:k")}

    def test_unrelated_chain_is_excluded_even_from_an_endpoint(self):
        nodes, _ = variable_neighborhood(self.NODES, self.EDGES, "var:a")
        ids = {n["id"] for n in nodes}
        assert "var:x" not in ids and "var:y" not in ids
        # a's descendants pull in the whole chain through it.
        assert {"var:a", "var:b", "var:c"} <= ids
