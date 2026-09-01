"""Knowledge-graph strategic insight: interpret the curated KG for a brand.

The scoping helpers here are a server-side port of the /knowledge-graph page's
client-side pipeline (frontend/src/pages/KnowledgeGraph.tsx: causalGoldStandardGraph,
dedupeParallelEdges, variableNeighborhoodGraph). The insight must be grounded in
the SAME graph the analyst is looking at — same type filters, same brand scoping,
same parallel-edge collapse, same variable neighborhood — or the narrative and the
canvas silently disagree.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

# Page-parity fetch scope: exactly what the page requests from /graph/nodes and
# /graph/relationships (its CAUSAL_NODE_TYPES / CAUSAL_REL_TYPES constants and
# NODE_FETCH_LIMIT / REL_FETCH_LIMIT = 2000, the backend cap).
PAGE_ENTITY_TYPES = ["Variable", "KPI", "CausalPath", "Region"]
PAGE_RELATIONSHIP_TYPES = ["CAUSES", "EXPLAINS", "INFLUENCES", "AFFECTS"]
PAGE_FETCH_LIMIT = 2000

# NOTE on shapes: SemanticMemory.list_relationships flattens edge properties into
# the top level of each dict (brand/region/confidence are direct keys), unlike the
# frontend's nested ``properties`` object. These helpers take that flat shape.


def dedupe_parallel_edges(relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-(brand x region) physical copies of each logical edge.

    The gold-standard sync MERGEs one edge per {brand, region}, so a single
    ``a -CAUSES-> b`` appears up to 3 brands x 4 regions = 12 times. Keyed by
    (source, type, target); keeps the highest-confidence copy as representative,
    surfacing the union of brands/regions and a ``parallel_edge_count``.
    Counting the copies as distinct relationships would overstate the edge
    profile the analyst sees by up to 12x.
    """
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in relationships:
        key = (str(r.get("source_id")), str(r.get("type")), str(r.get("target_id")))
        entry = by_key.get(key)
        if entry is None:
            entry = {"edge": r, "brands": set(), "regions": set(), "count": 0}
            by_key[key] = entry
        entry["count"] += 1
        if isinstance(r.get("brand"), str):
            entry["brands"].add(r["brand"])
        if isinstance(r.get("region"), str):
            entry["regions"].add(r["region"])
        if float(r.get("confidence") or 0) > float(entry["edge"].get("confidence") or 0):
            entry["edge"] = r
    deduped = []
    for entry in by_key.values():
        edge = {k: v for k, v in entry["edge"].items() if k not in ("brand", "region")}
        if entry["brands"]:
            edge["brands"] = sorted(entry["brands"])
        if entry["regions"]:
            edge["regions"] = sorted(entry["regions"])
        if entry["count"] > 1:
            edge["parallel_edge_count"] = entry["count"]
        deduped.append(edge)
    return deduped


def causal_gold_standard_graph(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    brand: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive the causal gold-standard graph the page renders for ``brand``.

    'All' keeps every causal edge; a brand keeps its tagged edges (matched
    case-insensitively — the graph holds 'Kisqali' AND 'kisqali' dupes) plus the
    untagged brand-agnostic structural edges. Then only nodes touched by >=1
    kept edge survive (no isolated singletons), and parallel copies collapse.
    """
    target = brand.lower()
    kept = [
        r
        for r in relationships
        if brand == "All" or not isinstance(r.get("brand"), str) or r["brand"].lower() == target
    ]
    touched = {r.get("source_id") for r in kept} | {r.get("target_id") for r in kept}
    kept_nodes = [n for n in nodes if n.get("id") in touched]
    node_ids = {n.get("id") for n in kept_nodes}
    surviving = [
        r for r in kept if r.get("source_id") in node_ids and r.get("target_id") in node_ids
    ]
    return kept_nodes, dedupe_parallel_edges(surviving)


def variable_neighborhood(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    variable_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Narrow the (brand-scoped) graph to one variable's causal neighborhood:
    every ancestor and descendant along CAUSES edges (the full chains through
    it) plus the structural context — non-CAUSES edges touching those nodes,
    and their far endpoints."""
    fwd: dict[str, list[str]] = {}
    rev: dict[str, list[str]] = {}
    for r in relationships:
        if r.get("type") != "CAUSES":
            continue
        fwd.setdefault(str(r.get("source_id")), []).append(str(r.get("target_id")))
        rev.setdefault(str(r.get("target_id")), []).append(str(r.get("source_id")))

    def reach(start: str, adj: dict[str, list[str]]) -> set[str]:
        seen = {start}
        queue = [start]
        while queue:
            for nxt in adj.get(queue.pop(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    core = reach(variable_id, fwd) | reach(variable_id, rev)
    context: set[str] = set()
    for r in relationships:
        if r.get("type") == "CAUSES":
            continue
        if r.get("source_id") in core:
            context.add(str(r.get("target_id")))
        elif r.get("target_id") in core:
            context.add(str(r.get("source_id")))
    keep = core | context
    kept_nodes = [n for n in nodes if n.get("id") in keep]
    kept_rels = [
        r
        for r in relationships
        if (
            r.get("source_id") in core and r.get("target_id") in core
            if r.get("type") == "CAUSES"
            else (r.get("source_id") in core or r.get("target_id") in core)
            and r.get("source_id") in keep
            and r.get("target_id") in keep
        )
    ]
    return kept_nodes, kept_rels


try:
    import dspy

    class KnowledgeGraphInsightSignature(dspy.Signature):
        """Interpret a curated pharmaceutical knowledge graph for a brand analyst,
        STRICTLY grounded in the provided counts and entity names. Use ONLY the
        numbers and names given; NEVER invent nodes, edges, or confidence values.
        Explain what the structure implies about causal drivers/levers; if the graph
        is sparse, say so plainly rather than over-reading it.

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers; plain numbered enumeration like "1." is fine."""

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
