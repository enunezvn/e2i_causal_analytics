# Performance benchmark synthetic data curation

Curated for issue #391 post-implementation checklist (PERFORMANCE slice). See
the parent CURATION.md for the retrieval-quality curation policy; this file
documents the performance-benchmark synthetic data harness.

## Files

| File | Purpose |
|------|---------|
| `synthetic_graph.jsonl` | Synthetic provenance-DAG edges for cascade-latency benchmark (Box 1). |
| `retrieval_queries.jsonl` | Reused from issue #377 for HybridRetriever-latency benchmark (Box 2). |
| `synthetic_corpus.jsonl` | Synthetic corpus for BM25-index-build benchmark (Box 3). |

## `synthetic_graph.jsonl` schema

Each non-comment line is a JSON object representing a single directed edge in
the provenance DAG. The cascade-latency benchmark loads these into a
``_FakeSupabaseGraph`` stub that approximates the production
``insight_edges`` table shape, then exercises the real
``src.memory.lifecycle.invalidator.cascade_invalidate`` BFS via dependency
injection.

```json
{
  "source_type": "causal_path",
  "source_id": "cp-0001",
  "target_type": "trigger",
  "target_id": "trg-0001",
  "brand": "kisqali"
}
```

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `source_type` | string | yes | Parent node type (e.g. `causal_path`, `trigger`) |
| `source_id` | string | yes | Parent node id |
| `target_type` | string | yes | Child node type (e.g. `trigger`, `ml_prediction`, `executive_insight`) |
| `target_id` | string | yes | Child node id |
| `brand` | string | yes | Brand scope of the edge (`kisqali`, `fabhalta`, ..., or `all`) |

## Synthetic graph generation policy

We ship a synthetic graph of N=1000 nodes / ~5000 edges arranged so that the
deepest BFS path from the seed root has 5 hops — matching issue #391's
"< 500ms for 5-hop BFS" target. The graph is structured as:

* 1 seed root (`causal_path:cp-root`)
* Hop 1: 10 children (mix of `trigger` / `ml_prediction` / `executive_insight`)
* Hop 2: 50 grandchildren
* Hop 3: 200
* Hop 4: 500
* Hop 5: 239

All edges are brand-scoped to a single brand (`bench`) so the BFS does not
short-circuit at brand boundaries.

This is a synthetic-baseline benchmark per the placeholder-first-run policy:
real production graph topology may not match this synthetic shape and the
first run on a given environment BLESSES its measured baseline. Subsequent
runs compare against the blessed value with the documented tolerance band.

## `synthetic_corpus.jsonl` schema

Each non-comment line is a JSON object representing a single document for the
BM25-rebuild benchmark.

```json
{
  "doc_id": "doc-bench-0001",
  "content": "Kisqali TRx growth West region Q3 confidence score 0.85 ..."
}
```

We ship ~1500 synthetic docs (~50 words each, ~75k tokens total). The benchmark
slices to N=1000 / 5000 / 10000 doc-equivalents by repetition and measures
build wall-clock at each slice — to produce a curve, not a single point.

## Tolerances

Tolerances are defined alongside each benchmark in
`tests/benchmarks/baselines/performance.json`. See the per-test docstrings for
the per-box target + tolerance shape.
