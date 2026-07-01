from src.insights.knowledge_graph import build_grounding, generate_insight

NODES = [
    {"id": "1", "name": "Adherence", "type": "Variable"},
    {"id": "2", "name": "NRx", "type": "KPI"},
    {"id": "3", "name": "Copay", "type": "Variable"},
]
RELS = [
    {"source_id": "3", "target_id": "1", "type": "CAUSES", "confidence": 0.82},
    {"source_id": "1", "target_id": "2", "type": "CAUSES", "confidence": 0.77},
]


def test_build_grounding_counts_and_chips():
    g = build_grounding("Kisqali", NODES, RELS, node_count=3, rel_count=2)
    assert "Variable" in g["node_summary"] and "KPI" in g["node_summary"]
    assert any(c["label"] == "Nodes" and c["value"] == "3" for c in g["grounding"])
    assert any(c["label"] == "Relationships" and c["value"] == "2" for c in g["grounding"])


def test_generate_insight_fallback_is_grounded_without_lm():
    g = build_grounding("Kisqali", NODES, RELS, node_count=3, rel_count=2)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "3" in out["insight"] and "Kisqali" in out["insight"]  # real numbers, no fabrication
    assert isinstance(out["key_takeaways"], list)
