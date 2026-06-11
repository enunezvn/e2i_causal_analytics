# Faithful probe evidence — cognitive RAG hybrid search (2026-06-08)

Environment: prod droplet `e2i-analytics-prod`. Live Supabase pgvector (:5433/:6543),
FalkorDB (:6381), real Claude via DSPy LM `anthropic/claude-sonnet-4-20250514`. NO MOCKS.
Pipeline under test: `src/api/routes/chatbot_dspy.py::cognitive_rag_retrieve` (the live
path called by `chatbot_graph.retrieve_rag_node`, default-enabled `CHATBOT_COGNITIVE_RAG=true`).

## Probe 1 — full pipeline, 3 query types (probe_pipeline.py)

For ALL three queries (KPI / explanation / context), only DENSE + SPARSE fired in the
cognitive path; GRAPH never fired. Control basic `hybrid_search(kpi_name='trx')` DID fire GRAPH.

| query | rewrite method | backends fired | DENSE n | SPARSE n | scoring LLM calls | evidence kept | total latency |
|---|---|---|---|---|---|---|---|
| "TRx trend for Kisqali in Northeast" | dspy | DENSE, SPARSE | 6 | 0 | 5 (sequential) | 1 | 17.4 s |
| "Why did Kisqali adoption increase…" | dspy | DENSE, SPARSE | 6 | 0 | 5 (sequential) | 1 | 17.3 s |
| "Summarize recent activity for Fabhalta" | dspy | DENSE, SPARSE | 6 | 0 | 5 (sequential) | 1 | 18.2 s |
| CONTROL basic hybrid_search(kpi_name='trx') | n/a | DENSE, SPARSE, **GRAPH** | 6 | 0 | 0 | 5 returned | **1.3 s** |

Latency breakdown (cognitive): rewrite ≈ 6.6–7.2 s (1 Claude call) + retrieval ≈ 0.3–2.7 s
+ evidence scoring ≈ 14.7–17.9 s (5 sequential Claude calls, each ≈ 3 s).

The SAME generic procedural row won for all 3 distinct queries:
`[PROC] composition_comp_571cb03a: Needs causal then regional analysis` (rel=0.3, rrf≈0.0082).
`graph_entities` were extracted by the rewrite (e.g. ['Kisqali','Northeast','TRx','QTD',…])
but never passed to `hybrid_search` (kpi_name=None hardcoded) → graph leg dead.

## Probe 2 — corpus / sparse / scoring (probe_corpus_sparse_scoring.py)

Corpus row counts:
- episodic_memories = 72
- procedural_memories = 1322
- agent_activities = 2323
- causal_paths = 50
- triggers = 4356

DENSE results for a KPI query (top-8, vec cosine score, DSPy relevance when DSPy configured):
```
[0] vec 0.811  Causal analysis: hcp_engagement_level -> patient_conversion_rate, ATE=0.413 …
[1] vec 0.811  (DUPLICATE of [0] — different id, survives source_id dedup)
[2] vec 0.779  [PROC] composition_comp_571cb03a: Needs causal then regional analysis
[3] vec 0.778  [PROC] composition_comp_8c934339: Three independent analyses
[4] vec 0.752  Feature Analysis: tier0_e2e_9bd2c46f. Top features: N/A. Selected 0 features.
[5] vec 0.749  [PROC] composition_comp_8bf7009f: Sequential analysis chain
[6] vec 0.732  [PROC] composition_comp_0f7cd757: Simple two-part question
[7] vec 0.726  QC Report: passed. Score: 1.00. Gate: PASSED. Leakage: none.
```
None of these mention TRx / Kisqali / Northeast. The vector store (episodic+procedural) holds
AGENT-INTERNAL bookkeeping, not KPI/commercial/explanatory content.

SPARSE / fulltext: `hybrid_fulltext_search` returns 0 rows for EVERY term tested
('Kisqali','TRx','adoption','agent', phrase queries). Direct RPC call also 0.

NOTE: in probe 2, `score_evidence_dspy` returned a uniform 0.5 fallback because DSPy LM was
not configured in that process (no prior rewrite call). `score_evidence_dspy` does NOT call
`_ensure_dspy_configured()` itself — it relies on a prior rewrite having configured DSPy.
In probe 1 (rewrite ran first) the scorer returned real values (0.1–0.3).

## SPARSE root cause (in-database, read-only psql)

`search_vector` is fully populated (agent_activities 2323/2323, triggers 4356/4356,
causal_paths 50/50) — NOT an empty-column problem.

But the tsvector is built from agent category tokens, e.g.:
`'causal':1A 'impact':2A 'recommend':3B 'insight':4C 'found':5C 'anomali':7C 'detect':8C`
`'gap':1A 'analysi':3B 'analyz':2A`

Match tests:
- `agent_activities WHERE search_vector @@ websearch_to_tsquery('english','agent')` → 0
- `triggers WHERE search_vector @@ websearch_to_tsquery('english','kisqali')` → 0

→ Domain terms (brand, KPI, region, free text) are NOT in the indexed tsvector, so fulltext
matches nothing for real queries. RPC defined in `database/memory/022_hybrid_search_max_staleness.sql`
(searches causal_paths/agent_activities/triggers `.search_vector`).

## Net (production reality of the "hybrid search across 3 backends")
- GRAPH: never wired in the cognitive path (and causal_paths only 50 rows).
- SPARSE: returns 0 for all domain queries (tsvector indexes wrong tokens).
- DENSE: only contributor; covers episodic(72)+procedural(1322) agent-internal memories only.
→ RRF fuses a single non-empty list; "hybrid" + RRF are effectively inert. The corpus the
pipeline can reach does not contain KPI/explanation/context answers.
