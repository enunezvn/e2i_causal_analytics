# Retrieval benchmark query-set curation

Curated for issue #377 (memory-subsystems: Phase 2 benchmark harness — Recall@10
+ MRR for HybridRetriever). See plan `.claude/plans/e2i_memory_subsystems_implementation_plan.md`
§"Phase 2 — Remaining" line 74 + §"Recommended sequencing" item 4.

## Source of truth

The labeled query-set lives at `retrieval_queries.jsonl` in this directory.
Each non-comment line is a JSON object with the schema documented below.

## Schema

```json
{
  "query_id": "Q-001",
  "query_text": "What is driving Kisqali TRx growth in the West region?",
  "relevant_doc_ids": ["doc-causal-001", "doc-trigger-014"],
  "relevance_grades": {"doc-causal-001": 3, "doc-trigger-014": 2},
  "category": "brand-scoped",
  "expected_sources": ["causal_paths", "triggers"],
  "tier3_consumer": "drift_monitor",
  "filters": {"agent_name": "drift_monitor"},
  "max_staleness": 0.0,
  "notes": "Optional human-readable rationale for the relevance judgments."
}
```

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `query_id` | string | yes | Stable identifier for cross-baseline comparison |
| `query_text` | string | yes | The natural-language query passed to `HybridRetriever.search` |
| `relevant_doc_ids` | list[string] | yes | Expected-relevant `source_id` values for Recall@10 / MRR |
| `relevance_grades` | dict[string, int] | optional | doc_id → integer grade (3/2/1) for nDCG@10 |
| `category` | string | yes | One of: `brand-scoped`, `kpi-scoped`, `entity-grounded`, `mixed-source` |
| `expected_sources` | list[string] | yes | Subset of `{causal_paths, agent_activities, triggers, episodic_memories, procedural_memories, semantic_graph}` |
| `tier3_consumer` | string | yes | Which Tier 3 agent's defaults to apply: `drift_monitor`, `experiment_designer`, `experiment_monitor`, `health_score` |
| `filters` | dict | yes | Filters passed to `HybridRetriever.search`; matches the Tier 3 consumer defaults from PR #374 (issue #373) |
| `max_staleness` | float, nullable | yes | Staleness ceiling; matches Tier 3 defaults (`0.0` for all consumers except `health_score`, which passes filters through but no agent_name default — see `src/rag/retriever.py:247-255` for binary semantics) |
| `notes` | string | optional | Free-text rationale; not consumed by the harness |

`query_id` MUST be unique within the file. The harness rejects duplicate
ids at load time.

## Coverage requirements (from issue #377 scope §A)

The query-set must cover:

1. **Brand-scoped queries** — Remibrutinib (CSU), Fabhalta (PNH), Kisqali
   (HR+/HER2- breast cancer). Brand identifiers verified against
   `src/rag/entity_extractor.py:67` canonical list.
2. **KPI-scoped queries** — TRx, NRx, conversion_rate, market_share,
   adoption_rate. KPI identifiers verified against
   `src/rag/entity_extractor.py:101` and `src/rag/causal_rag.py:196`.
3. **Entity-grounded queries** — specific HCP / patient / cohort IDs.
4. **Mixed-source queries** — expected hits in two or more of
   `{causal_paths, agent_activities, triggers}` so the RRF fusion at
   `src/rag/retriever.py:263` is exercised across the three streams
   (dense → episodic/procedural, sparse → causal_paths/agent_activities/
   triggers, graph → semantic_graph).

## Query-count rationale (≥30)

Issue #377 DoD specifies "≥30 labeled queries". The plan body §Step 2.7
(line 644) specifies "20+ labeled query-result pairs" — the issue's ≥30
target is the stricter of the two, so we adopt it.

We add a few queries above the floor (the file ships with 36) so removing a
miscategorised query in audit doesn't drop us below the gate. We do NOT
claim ≥30 as a statistically-derived sample-size threshold — it is simply
the issue's DoD number quoted verbatim. (Per memory
`feedback_overclaiming_during_planning`: do not invent thresholds.)

Distribution across the four required categories:

| Category | Count |
|----------|-------|
| brand-scoped | 9 |
| kpi-scoped | 9 |
| entity-grounded | 9 |
| mixed-source | 9 |
| **total** | **36** |

Across Tier 3 consumers:

| Consumer | Count |
|----------|-------|
| drift_monitor | 9 |
| experiment_designer | 9 |
| experiment_monitor | 9 |
| health_score | 9 |

## Synthetic vs. production-log queries

This file holds **synthetic + curated** queries only. Real-data labeled
queries from production logs are explicitly **out of scope** per issue
#377 — see the issue body §"Out of scope":

> Real-data labeled queries from production logs — defer until synthetic +
> curated covers ≥80% of expected query patterns.

That ≥80% gate is the issue author's; the harness does not measure pattern
coverage today. Treat the synthetic corpus as the v1 baseline; expect a
follow-up issue when production-log telemetry is available.

## Relevance-judgment methodology

For each `(query, doc_id)` pair claimed relevant, we apply the following
checklist (binary; doc is "relevant" iff all three pass):

1. **On-topic** — the document's `content` (per `RetrievalResult.content`)
   would plausibly help answer the query as posed.
2. **Source-aligned** — the document's `source` field matches one of the
   query's `expected_sources`.
3. **Tier-3-consumable** — the document would not be filtered out by the
   `filters` and `max_staleness` parameters that the named `tier3_consumer`
   would pass to `HybridRetriever.search` per the wire-in established by
   PR #374 (issue #373).

For nDCG@10 grades (optional, only set on a subset of queries):

| Grade | Meaning |
|-------|---------|
| 3 | Highly relevant — would be the primary citation for an answer |
| 2 | Relevant — would be cited as supporting evidence |
| 1 | Marginally relevant — useful context but not load-bearing |

(Scale matches the standard TREC graded-relevance convention; see
Manning IIR §8.5.1.)

## Limitations + known gaps

- **No live-retrieval verification at curation time** — the relevance
  judgments are based on the document IDs we expect the harness to see
  given the wired Tier 3 defaults. If the corpus shape changes (new
  source tables, new staleness semantics), the labels must be re-audited.
- **Synthetic doc IDs** — the `doc-causal-NNN` / `doc-trigger-NNN` / etc.
  identifiers in this file are synthetic placeholders. The harness only
  asserts on aggregate metrics; per-query Recall@10 against a real
  retriever will be 0.0 until the test fixture corpus is seeded with
  documents carrying those exact `source_id`s. This is intentional — the
  v1 harness shape lands first (issue #377), corpus seeding lands in a
  follow-up issue (see PR body).
- **No domain-SME review** — these are curator-generated queries, not
  vetted by a commercial-analytics SME. The plan does not specify an
  SME-review requirement, and the issue body explicitly defers
  production-log curation. Mark queries as "preliminary" in the notes
  field rather than treating them as ground truth.

## Updating the query-set

When changing this file:

1. Run `pytest tests/benchmarks/_metrics_test.py -v` to confirm metric
   primitives still green.
2. Run `pytest -m benchmark tests/benchmarks/test_retrieval_quality.py -v
   -p no:xdist -o "addopts="` to see how the change moves aggregate
   metrics against the baseline at `tests/benchmarks/baselines/retrieval_quality.json`.
3. If aggregate Recall@10 or MRR shifts by more than the configured
   tolerances (see workflow YAML), explicitly re-bless the baseline in the
   same PR — do not adjust tolerances to hide the shift.
4. Bump the file header `version` field so old baselines can be detected
   and rejected.
