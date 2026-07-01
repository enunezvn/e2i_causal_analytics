"""Knowledge-graph strategic insight: interpret the curated KG for a brand."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class KnowledgeGraphInsightSignature(dspy.Signature):
        """Interpret a curated pharmaceutical knowledge graph for a brand analyst,
        STRICTLY grounded in the provided counts and entity names. Use ONLY the
        numbers and names given; NEVER invent nodes, edges, or confidence values.
        Explain what the structure implies about causal drivers/levers; if the graph
        is sparse, say so plainly rather than over-reading it."""

        scope: str = dspy.InputField(desc="Brand/region scope of this graph view")
        node_summary: str = dspy.InputField(desc="Node counts by type and total")
        top_hubs: str = dspy.InputField(desc="Highest-degree entities: name, type, degree")
        key_paths: str = dspy.InputField(desc="Notable CAUSES/INFLUENCES chains + confidence")
        edge_summary: str = dspy.InputField(desc="Relationship counts by type + confidence range")

        interpretation: str = dspy.OutputField(
            desc="What the structure says about causal drivers/levers for this brand"
        )
        key_takeaways: list = dspy.OutputField(desc="3-5 specific, grounded takeaways")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    KnowledgeGraphInsightSignature = None  # type: ignore[assignment,misc]


def build_grounding(
    scope: str,
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    node_count: int,
    rel_count: int,
) -> dict[str, Any]:
    node_types: Counter = Counter(n.get("type", "Unknown") for n in nodes)
    # The type breakdown is computed over the fetched `nodes` (a bounded sample);
    # label it as "analyzed" so the sum stays consistent, and note the true taxonomy
    # total separately when it is larger (never present a breakdown that sums to a
    # different number than the headline — that reads as fabricated).
    analyzed = len(nodes)
    total_note = (
        f" (of {node_count} in the shared node taxonomy)"
        if isinstance(node_count, int) and node_count > analyzed
        else ""
    )
    node_summary = f"{analyzed} nodes analyzed{total_note}: " + ", ".join(
        f"{t}={c}" for t, c in node_types.most_common()
    )
    degree: Counter = Counter()
    name_by_id = {n.get("id"): n.get("name", n.get("id")) for n in nodes}
    type_by_id = {n.get("id"): n.get("type", "Unknown") for n in nodes}
    for r in relationships:
        degree[r.get("source_id")] += 1
        degree[r.get("target_id")] += 1
    top_hubs = (
        "; ".join(
            f"{name_by_id.get(nid, nid)} ({type_by_id.get(nid, '?')}, degree {d})"
            for nid, d in degree.most_common(5)
        )
        or "none"
    )
    key_paths = (
        "; ".join(
            f"{name_by_id.get(r.get('source_id'), r.get('source_id'))} -{r.get('type')}-> "
            f"{name_by_id.get(r.get('target_id'), r.get('target_id'))} "
            f"(conf {float(r.get('confidence') or 0):.2f})"
            for r in relationships[:6]
        )
        or "none"
    )
    edge_types: Counter = Counter(r.get("type", "?") for r in relationships)
    confs = [float(r.get("confidence") or 0) for r in relationships if r.get("confidence")]
    # Edges are scoped (the route filters relationships to the brand when one is
    # selected) while nodes are the shared/global taxonomy — make that explicit so a
    # brand view with few edges is not misread as a globally sparse graph.
    edge_summary = f"{rel_count} relationships in scope ({scope}): " + ", ".join(
        f"{t}={c}" for t, c in edge_types.most_common()
    )
    if confs:
        edge_summary += f"; confidence {min(confs):.2f}-{max(confs):.2f}"
    return {
        "scope": scope,
        "node_summary": node_summary,
        "top_hubs": top_hubs,
        "key_paths": key_paths,
        "edge_summary": edge_summary,
        "grounding": [
            {"label": "Nodes", "value": str(analyzed)},
            {"label": "Relationships", "value": str(rel_count)},
            {"label": "Node types", "value": str(len(node_types))},
        ],
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"For {g['scope']}, the curated graph holds {g['node_summary']}. "
        f"Highest-connectivity entities: {g['top_hubs']}. "
        f"Key causal links: {g['key_paths']}. "
        f"Edge profile: {g['edge_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["node_summary"], f"Top hubs: {g['top_hubs']}"],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    """LLM interpretation grounded in ``g``, or a deterministic factual fallback."""
    pred = run_signature(
        KnowledgeGraphInsightSignature,
        scope=g["scope"],
        node_summary=g["node_summary"],
        top_hubs=g["top_hubs"],
        key_paths=g["key_paths"],
        edge_summary=g["edge_summary"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }
