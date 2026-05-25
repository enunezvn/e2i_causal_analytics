# Design: Local pg+pgvector substrate for the HybridRetriever latency regression guard

**Issue:** #414 — re-bless HybridRetriever p50/p95 baselines
**Date:** 2026-05-25
**Status:** Approved (design); implementation plan pending

---

## 1. Background & problem

`tests/benchmarks/baselines/performance.json` holds p50/p95 latency baselines for six
benchmark "boxes". PR #413 (closing #403) CI-blessed four of them. The two
HybridRetriever baselines — `hybrid_retriever_search_p50` and
`hybrid_retriever_search_p95` — were intentionally left at `0.0` placeholders because
`tests/benchmarks/test_hybrid_retriever_latency.py` **skips** in CI without
`SUPABASE_URL` + `SUPABASE_KEY` + `OPENAI_API_KEY`.

Issue #414 originally proposed unblocking by adding those three secrets to CI and
re-blessing from a live end-to-end run. On review we **rejected that premise**:

- The benchmark calls `src.rag.retriever.hybrid_search()` end-to-end with no substrate
  injection, so its wall-clock is dominated by **one live OpenAI embedding call + two
  remote Supabase round-trips** — i.e. it measures third-party API/network latency, not
  our code.
- As a CI **regression guard** that is noisy (OpenAI tail-latency variance), toothless
  (a real code regression is swamped by network jitter, forcing wide tolerance bands),
  costly (paid calls every run), and flaky.

**Decision:** this baseline is a *code-latency regression guard*, not a live-SLA
monitor. We therefore refactor it to run against a **local, deterministic, secret-free
pg+pgvector substrate** that isolates our code while exercising the real SQL +
pgvector index path.

## 2. Goal & scope

**Goal:** convert the two hybrid baselines from `0.0` placeholders into a reproducible,
secret-free, in-CI regression guard, blessed from CI medians per the #413 methodology.

**In scope**
- A local Postgres+pgvector substrate, seeded with a fixed corpus, that
  `hybrid_search` runs against.
- The two latency baselines + their tolerance bands + the drift-guard meta-test +
  benchmark docstring.

**Out of scope**
- The #377 recall@10/MRR *quality* re-bless (it can reuse this substrate later, with a
  real embedder — noted, not built here).
- FalkorDB: the latency path never invokes the graph stream (see §5).
- PostgREST / running supabase-py against a local endpoint (see §4 decision).

## 3. Key findings that shape the design

1. **No prior art.** `retrieval-benchmarks.yml` (the recall/MRR quality benchmark for
   the *same* `HybridRetriever`, #377) has the identical problem — it skips without live
   services and its baseline is also unblessed. It does **not** stand up a pgvector
   container. So the substrate built here would unblock both #414 and #377 → build it as
   reusable test infra.
2. **The latency benchmark never touches the graph stream.** `_run_one_query_timed`
   calls `hybrid_search(query, k, filters, max_staleness)` with no `entities`/`kpi_name`,
   so `HybridRetriever.search` leaves `graph_results = []` (`src/rag/retriever.py:305-307`).
   No FalkorDB needed.
3. **The connector reaches Postgres only via supabase-py `.rpc()`**
   (`src/rag/memory_connector.py:143,247` → `get_supabase_client()` → PostgREST). A bare
   pgvector container is not drop-in for the production data-access path.
4. **The SQL functions are plain SQL.** `hybrid_vector_search` and
   `hybrid_fulltext_search` are `CREATE OR REPLACE FUNCTION` definitions in
   `database/memory/011_hybrid_search_functions_fixed.sql` and `022_…`. They load into a
   bare Postgres and can be invoked via `SELECT * FROM <fn>(...)`.
5. **Injection seam exists.** `get_memory_connector()` is a module-level singleton
   (`_memory_connector`) with a `reset_memory_connector()` (`memory_connector.py:480-496`).
   The retriever resolves the connector per-call, so monkeypatching the global is
   sufficient — no production setter required.
6. **A deterministic embedder seam exists.** `src/memory/services/factories.py` defines
   an `EmbeddingService` ABC with `OpenAIEmbeddingService` / `LocalEmbeddingService`
   implementations.

## 4. Architecture

### Injection (zero production change)
The benchmark fixture monkeypatches `src.rag.memory_connector._memory_connector` with a
test-side connector and restores via the existing `reset_memory_connector()` in
teardown. No production code is modified.

### `_DirectSQLMemoryConnector` (test-side, `tests/benchmarks/`)
Implements the four methods the retriever calls:
- `vector_search_by_text(query_text, k, filters, min_similarity, max_staleness)` —
  embeds via the deterministic embedder, then `SELECT * FROM hybrid_vector_search(...)`
  via `psycopg` against the local DB.
- `fulltext_search(query_text, k, filters, max_staleness)` —
  `SELECT * FROM hybrid_fulltext_search(...)`.
- `graph_traverse(...)`, `graph_traverse_kpi(...)` — return `[]` (never called here;
  present to satisfy the interface).

Calling the **same SQL functions** means the real pgvector index traversal and SQL plan
are exercised; only the thin, stable supabase-py/PostgREST HTTP shim (not our code) is
bypassed. This also yields lower run-to-run variance → a more stable regression guard.

### Deterministic embedder
A fixed hash→1536-dim normalized-vector embedder (zero downloads, fully reproducible).
pgvector top-k latency does not depend on embedding *quality*, so this is faithful for a
latency guard. The corpus is embedded with the same function at seed time. (A real
embedder would only be swapped in if #377 quality reuses this substrate.)

## 5. Data flow (one benchmark query)

```
hybrid_search(query)
  → HybridRetriever.search (dense + sparse concurrent; graph short-circuits empty)
    → injected _DirectSQLMemoryConnector
      → deterministic embed(query)              # no OpenAI
      → SELECT * FROM hybrid_vector_search(...)  # local pgvector index
      → SELECT * FROM hybrid_fulltext_search(...) # local full-text
    → RRF fusion (our code)
  → timed wall-clock (perf_counter)
```

## 6. CI workflow changes (`benchmarks.yml`, Box 2 only)

- Add a health-gated `pgvector/pgvector:pg16` **service container**.
- Pre-Box-2 steps: `CREATE EXTENSION vector` → load minimal table DDL + the two SQL
  functions (`011_hybrid_search_functions_fixed.sql`, `022_…`) → seed the fixed corpus.
- Box 2 runs with `BENCH_SUBSTRATE=local_pg` + `BENCH_PG_DSN` pointing at the container.
  **No secrets, no OpenAI, no Supabase.** The push/weekly-cron triggers now run it for
  free (the earlier cost concern was contingent on live APIs and evaporates).

## 7. Test changes

- **Skip logic:** a single `_substrate_ready()` = (`BENCH_SUBSTRATE == "local_pg"`) OR
  (live services ready). The test runs under the local substrate instead of skipping;
  the `requires_supabase` / `sk-*` guards apply only to the legacy live path.
- **Re-bless:** ≥3 `workflow_dispatch` runs → median-of-p50s / median-of-p95s into
  `performance.json.mean_ms`; replace `_placeholder_rationale` with `_ci_observation` +
  `_blessed_from_ci_runs`; **derive tolerance bands from observed cross-run stdev**
  (replacing the placeholder 20%/25% rel + 50/100ms abs defaults).
- **Drift guard:** move both baselines from `_PLACEHOLDER_BOXES_HYBRID` →
  `_REBLESSED_BOXES` in `tests/unit/test_benchmarks_meta/test_baseline_no_placeholder.py`.
- **Docstring:** replace the "placeholder strategy retained for hybrid" language with
  "ci-blessed-median against local pg+pgvector substrate."

## 8. Error handling — fail closed, never silent-skip

If `BENCH_SUBSTRATE=local_pg` is set but the DB is unreachable, unseeded, or the SQL
functions are missing, the benchmark must **error**, not skip or bless `0.0`. Silent
degradation to empty/zero results is the exact #403 failure mode.

Because we inject the test-side `_DirectSQLMemoryConnector` (the production
`MemoryConnector` is not used in benchmark mode — preserving the zero-production-change
architecture), the fail-closed behavior lives in the test-side connector: it does **not**
swallow DB errors (no `except: return []`) and raises on connection/query failure. The
benchmark additionally treats an all-empty result set across the query set as a hard
failure, so a reachable-but-unseeded substrate also fails loudly rather than blessing a
fast-but-meaningless `0.0`.

## 9. Testing the substrate itself

Fast unit tests assert:
- seed → query returns non-empty results for known seeded queries;
- the deterministic embedder is stable across calls (same text → same vector);
- "broken substrate fails loudly" — pointing at an empty DB makes the benchmark error
  rather than bless `0.0`.

## 10. Open implementation details (resolved during planning, do not change this shape)

- Exact minimal table DDL set the two SQL functions depend on (episodic_memories,
  procedural_memories, causal_paths, agent_activities, triggers + their indexes).
- What already exists in `tests/benchmarks/data/` (query set vs corpus).
- pgvector index type (HNSW per README) and corpus size needed for stable, non-noise
  timings.

## 11. Follow-ups (not in this work)

- #377 retrieval-quality re-bless can reuse this substrate (with a real embedder).
- Optionally migrate `retrieval-benchmarks.yml` onto the shared substrate.
