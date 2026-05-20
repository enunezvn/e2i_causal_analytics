#!/usr/bin/env python3
"""Generate synthetic provenance-DAG JSONL for cascade-latency benchmark.

Produces ``tests/benchmarks/data/synthetic_graph.jsonl`` with a 5-hop DAG of
~1000 nodes / ~5000 edges rooted at ``causal_path:cp-root``. Each layer fans
out further than the last, so a BFS reaches depth 5 in one cascade run.

All edges are brand-scoped to ``bench`` so the cascade BFS does not
short-circuit at brand boundaries.

Re-run only if the graph shape needs to change (e.g., to add more depth or
test fan-out variance). The shipped JSONL is deterministic — re-running
produces identical output (seed pinned).

Usage::

    python scripts/benchmarks/gen_synthetic_graph.py

See ``tests/benchmarks/data/CURATION_PERF.md`` for the schema + curation
policy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Deterministic seed so the shipped JSONL is reproducible.
_SEED = 391_2026_05_20

# Per CURATION_PERF.md, the fan-out targets are:
_LAYER_SIZES = [1, 10, 50, 200, 500, 239]  # 1000 nodes total across 6 layers
_TARGET_TYPES = ("trigger", "ml_prediction", "executive_insight")
_BRAND = "bench"
_OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "benchmarks"
    / "data"
    / "synthetic_graph.jsonl"
)


def _node_id(layer_idx: int, idx_in_layer: int) -> tuple[str, str]:
    """Return (type, id) for a node at (layer, position).

    Layer 0 is the root (``causal_path``); deeper layers cycle through the
    invalidatable target types.
    """
    if layer_idx == 0:
        return ("causal_path", "cp-root")
    target_type = _TARGET_TYPES[idx_in_layer % len(_TARGET_TYPES)]
    # short id prefix per type so we can eyeball in the JSONL
    prefix = {"trigger": "trg", "ml_prediction": "mlp", "executive_insight": "exi"}[
        target_type
    ]
    return (target_type, f"{prefix}-L{layer_idx}-{idx_in_layer:04d}")


def main() -> None:
    rng = random.Random(_SEED)
    layers: list[list[tuple[str, str]]] = []
    for layer_idx, size in enumerate(_LAYER_SIZES):
        layer = [_node_id(layer_idx, i) for i in range(size)]
        layers.append(layer)

    edges: list[dict[str, str]] = []
    # For each non-root layer, every node has exactly one parent picked
    # uniformly at random from the previous layer. This keeps the DAG well-
    # connected (no orphans) and produces ~sum(layer_sizes[1:]) edges = ~999.
    # We add a small set of cross-layer edges so the BFS frontier expands
    # beyond strict tree shape — a real provenance DAG has multi-parent
    # nodes (per database/memory/021_insight_lifecycle.sql:228).
    for layer_idx in range(1, len(_LAYER_SIZES)):
        parents = layers[layer_idx - 1]
        children = layers[layer_idx]
        for child in children:
            parent = rng.choice(parents)
            edges.append(
                {
                    "source_type": parent[0],
                    "source_id": parent[1],
                    "target_type": child[0],
                    "target_id": child[1],
                    "brand": _BRAND,
                }
            )

    # Add ~4000 extra edges. Cross-layer edges are restricted to
    # immediately-adjacent layers (L → L+1) so BFS depth from root cannot
    # short-circuit past intermediate layers. With this restriction, the
    # shortest path from the root to any layer-5 node is EXACTLY 5 hops.
    #
    # Initial (codex iter-0 H1) implementation used ``rng.randrange(parent_layer
    # + 1, layer_count)``, which let root-to-deep shortcuts (e.g.
    # root → layer 3 directly) collapse the BFS depth distribution to
    # ``[1, 426, 528, 45, 0]`` — max depth 3, NOT 5. Restricting extras to
    # adjacent-layer edges preserves the 5-hop shortest-path invariant.
    extra_edges_target = 4000
    layer_count = len(_LAYER_SIZES)
    extras_added = 0
    while extras_added < extra_edges_target:
        parent_layer = rng.randrange(0, layer_count - 1)
        child_layer = parent_layer + 1  # adjacent-layer only (iter-1 fix)
        parent = rng.choice(layers[parent_layer])
        child = rng.choice(layers[child_layer])
        edges.append(
            {
                "source_type": parent[0],
                "source_id": parent[1],
                "target_type": child[0],
                "target_id": child[1],
                "brand": _BRAND,
            }
        )
        extras_added += 1

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Synthetic provenance-DAG for cascade-latency benchmark (issue #391, Box 1).\n"
        )
        fh.write("# See tests/benchmarks/data/CURATION_PERF.md for schema + policy.\n")
        fh.write(
            "# Deterministic generator: scripts/benchmarks/gen_synthetic_graph.py "
            f"(seed={_SEED}).\n"
        )
        fh.write(
            f"# Layers (root-first): {_LAYER_SIZES}, edges={len(edges)} (root + "
            f"per-layer parent edges + cross-layer extras).\n"
        )
        for edge in edges:
            fh.write(json.dumps(edge, separators=(",", ":")) + "\n")

    print(
        f"Wrote {len(edges)} edges across {sum(_LAYER_SIZES)} nodes to {_OUT_PATH}"
    )


if __name__ == "__main__":
    main()
