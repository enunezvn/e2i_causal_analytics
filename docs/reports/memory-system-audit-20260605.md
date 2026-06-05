# Memory System Audit — episodic / procedural / semantic_cache / cognitive_cycles

**Date:** 2026-06-05
**Branch:** `claude/bold-galileo-DULyN`
**Scope:** The pipeline's Memory subsystems — the 7 core memory tables
(`episodic_memories`, `procedural_memories`, `semantic_memory_cache`,
`cognitive_cycles`, `investigation_hops`, `learning_signals`,
`memory_statistics`) plus the lifecycle (consolidation/invalidation),
crystallization, sentinels, triple-stream retrieval, and DSPy-signal
peripherals that hang off them.

## Methodology (why you can trust the conclusions)

This audit followed the repo's own pinned directives — **REASON BEFORE
RULES** and **CHEAPEST-DISPROOF FIRST**. Every finding below was reached
by *disproving* a claim against ground truth, not by pattern-matching on
code shape.

- Five read-only sub-audits fanned out across the subsystems. Their raw
  output was treated as **hypotheses, not conclusions** — consistent with
  "a codex-ACCEPT plan is not authoritative until claims are independently
  verified."
- That skepticism was warranted: **8 of the agents' headline findings were
  false** and were rejected after verification (see
  [§5 Disproven claims](#5-claims-investigated-and-disproven)). Three
  separate agents independently missed *real* wiring (the DSPy signals
  writer, the procedure-outcome callers, the hybrid-search RPC callers) and
  one declared `cognitive_cycles` "fully wired" when it is orphaned.
- The strongest finding (the broken `016` RPC) is backed by an **executed
  programmatic cross-check** of the RPC's column references against the
  table DDL, not by eyeballing.

**Evidence scope / honesty:** This container is a fresh clone; the Python
test stack (`pytest` + the ~5 GB dependency set) is not installed, and no
live Postgres/Redis/FalkorDB is attached. The questions this audit answers
— *is there a producer/consumer?*, *do the RPC's columns exist?*, *is the
function ever called?* — are **static facts** whose faithful environment is
the source tree and SQL schema, which were verified directly. Claims that
would require a live runtime (e.g. "the RPC raises `UndefinedColumn` when
executed") are proven **at the schema level** (the column is absent from
the DDL) but were **not** executed against a live database. Mocked unit
tests would *not* catch the schema-drift defect — which is exactly why it
survived.

---

## 1. Bottom line up front

**The core of the memory system is genuinely wired and well-built.** The
hot path — episodic capture → consolidation/dedup → promotion →
crystallization → invalidation → triple-stream retrieval — is real, uses
real OpenAI embeddings (1536-dim, dimension-guarded), enforces brand
isolation at the DB level, and is exercised by the live agent pipeline
(`episodic_memory` is referenced across **22** agent files,
`procedural_memory` across **7**).

**The defects are concentrated in peripheral, mostly-orphaned scaffolding**,
not the hot path:

| # | Finding | Severity | Live harm today |
|---|---------|----------|-----------------|
| F1 | `cognitive_cycles` + `ConversationRepository` + `016` RPC are an **orphaned, superseded** trio; the RPC **references 7 columns that don't exist** | **MEDIUM** (latent correctness defect; misleading dead code) | **None** — unreachable (the only caller is never instantiated) |
| F2 | `semantic_memory_cache` sync wrappers are **orphaned** (never called, not on Celery beat) **and** the table has **no live reader** | **LOW–MEDIUM** | None — dormant both ends |
| F3 | `dspy_optimization_runs`, `dspy_prompt_versions`, `dspy_cognitive_context_history`, `investigation_hops` are **orphan tables** (zero `src/` references / no writer) | **LOW** | None — empty tables |
| F4 | `falkordb_synced` flag and `semantic_cache_ttl_minutes` config are **read but never acted on** (no sync-back, no TTL eviction) | **LOW** | None — unbounded growth only if F2 sync is ever wired |
| F5 | Cascade invalidation (`INVALIDATABLE_TABLES`) **excludes** `procedural_memories` & `semantic_memory_cache` | **INFO / by-design** | None — defensible; confirm intent |
| F6 | Docs don't distinguish the **orphaned** `cognitive_cycles` from the **live** `chatbot_conversations` conversation store | **LOW** (doc clarity) | None |

There is **no plan-354-style harmful silent mock** in the memory paths: the
one place that *looked* like fabricated output (the crystallizer's
deterministic narrative) turns out to be honest and self-labeled (see
[§5.6](#56-crystallizer-deterministic-prose-is-a-harmful-silent-mock--false)).

---

## 2. What is genuinely correct and well-built

These were verified positively and should **not** be touched:

- **Episodic capture & dedup.** `_compute_dedup_signature`
  (`consolidator.py`) is a deterministic SHA-256 over key fields with
  **brand always in the key**; a partial-unique index
  `(COALESCE(brand,''), dedup_signature)` (migration `026`) gives DB-level
  race safety. The consolidator's `run()` executes dedup **before**
  promotion, as the spec requires.
- **Promotion thresholds are config-driven, not hardcoded**
  (`SEMANTIC_MIN_CONFIRMATIONS=3`, `PROCEDURAL_MIN_USAGE=5`,
  `PROCEDURAL_MIN_SUCCESS_RATE`), overridable via the `Consolidator`
  constructor.
- **`procedural_memories` is fully wired.** Producers:
  `tool_composer` / `resource_optimizer` memory-hooks, `cognitive_rag`
  `store_procedure`, and consolidator template extraction. Outcome
  feedback: `update_procedure_outcome` is called from
  `feedback_learner/memory_hooks.py:193` **and** the `/memory` API route
  (`routes/memory.py:662`). Consumption: `success_rate` is a **DB
  `GENERATED` column** (`001`:261) and the `find_similar_procedures` RPC
  ranks by `similarity * (0.5 + 0.5 * success_rate)` (`001`:707).
- **Crystallization → `executive_insights`** writes 13 deterministic fields
  derived from real estimator/insight-edge/episodic state + 2 LLM-narrative
  fields gated by `E2I_CRYSTAL_LLM_NARRATIVES_ENABLED`; a partial-unique
  index prevents duplicate active crystals; Anthropic errors are
  narrow-caught so programming errors still surface.
- **Cascade invalidation** (`invalidator.py`) is a brand-scoped BFS over
  `insight_edges` with a visited-set cycle guard and best-effort Redis
  signalling; append-only provenance is enforced by DB triggers
  (migration `028`).
- **Triple-stream retrieval is wired.** `HybridRetriever` is constructed in
  `api/dependencies/rag.py:53` and `rag/retriever.py:419`; its backends
  call the **real** `hybrid_vector_search` / `hybrid_fulltext_search` RPCs
  via `rag/memory_connector.py:144,248` (RRF fusion, graph boost).
- **HPO warm-start is functional.** `hyperparameter_tuner.py:365` calls
  `study.enqueue_trial(warmstart_params)` — the documented Optuna
  warm-start API — and closes the loop with `record_warmstart_outcome`.
- **Service factories fail honestly.** Missing Supabase/Redis/FalkorDB env
  raises `ServiceConnectionError` rather than returning a silent fake; the
  embedding **dimension guard** rejects mismatched vectors before insert.
- **`semantic_memory_cache` itself is a real, correct table** (`001`:297)
  with a correct populating RPC (`sync_hcp_patient_relationships_to_cache`,
  `001b`). Its problem is purely that nothing calls the sync and nothing
  reads the table (F2), not that it's fake.

---

## 3. The "7 memory tables" claim is accurate

README/ONBOARDING say *"Memory (7): episodic_memories, procedural_memories,
semantic_cache, cognitive_cycles, etc."* That is **correct**: the core
schema `001_agentic_memory_schema_v1.3.sql` creates **exactly 7** tables —
`episodic_memories`, `procedural_memories`, **`semantic_memory_cache`**,
`cognitive_cycles`, `investigation_hops`, `learning_signals`,
`memory_statistics`. "`semantic_cache`" is shorthand for
`semantic_memory_cache`. (One sub-audit's "7 vs 14 doc-drift" claim
conflated these core tables with lifecycle/DSPy tables from *other*
migrations — rejected.)

---

## 4. Detailed findings

### F1 — `cognitive_cycles` trio is orphaned & superseded; the `016` RPC is broken  · MEDIUM

**What it is / intent.** Migration `016_conversation_similarity_search.sql`
("Enable vector similarity search on cognitive_cycles for RAG context")
plus `src/repositories/conversation.py` were built to use `cognitive_cycles`
as a conversation/query-history store with pgvector similarity and feedback.

**Why it's in this shape.** That design was **superseded** by
`ChatbotConversationRepository` (`table_name = "chatbot_conversations"`),
which is the store actually instantiated in the live route
`api/routes/copilotkit.py:945`. The live 4-phase cognitive workflow
(`cognitive_integration.py`) persists to `episodic_memories` +
`learning_signals` + FalkorDB instead — **never** to `cognitive_cycles`.

**Verified facts.**
- No `src/` code writes `cognitive_cycles` (no `insert/upsert/update`).
- `ConversationRepository` is **never instantiated** — only exported from
  `repositories/__init__.py`.
- The `016` RPCs are called **only** by `ConversationRepository`
  (`conversation.py:114,175`), which is unreachable.
- **Executed cross-check** of the RPC's `cc.<col>` references against the
  `cognitive_cycles` DDL → **7 referenced columns do not exist**:
  `agent_response, created_at, feedback_at, feedback_score, feedback_text,
  feedback_type, response_type` (the table has `synthesized_response`,
  `confidence_score`, `started_at`/`completed_at`, and **no** `feedback_*`).
  **Verdict: the RPC would raise `UndefinedColumn` if ever executed.**

**Harm now.** Nil — nothing reachable calls it. But it is a real latent
correctness defect and misleading dead code: a future engineer wiring
`ConversationRepository` would hit a runtime crash. The RPC was evidently
written against a *chatbot-with-feedback* mental model that never matched
the *4-phase-cycle* table it targets.

**Classification:** REWIRE-or-RETIRE (recoverable intent exists — do not
delete blindly). Either (a) retire the superseded trio (`conversation.py`,
the `016` RPCs, and — if confirmed unused — the `cognitive_cycles` /
`investigation_hops` tables), or (b) if conversation-history-on-
`cognitive_cycles` is still wanted, fix the RPC columns to match the DDL and
wire a producer. Confirm intent before either.

### F2 — `semantic_memory_cache` is dormant on both ends  · LOW–MEDIUM

`sync_data_layer_to_semantic_cache` (`semantic_memory.py:1579`) and
`sync_treatment_relationships_to_cache` (`episodic_memory.py:886`) wrap a
**working** RPC, but have **zero callers** in `src/` and **no Celery-beat
entry**. Separately, **no live code reads** `semantic_memory_cache` (only a
config table-name constant references it). So the producer never runs and
there is no consumer — the table is dormant scaffolding for a Supabase
hot-cache mirror of FalkorDB relationships that was never connected at
either end.

**Harm now.** Nil (no reader ⇒ no wrong answers). **Classification:**
KEEP-AS-INTENTIONAL-PLACEHOLDER **or** REWIRE — if the cache is on the
roadmap, add a beat entry for the sync and wire the graph path to read it;
if not, retire the wrappers and document the table as deploy-seed-only.

### F3 — Orphan tables  · LOW

Zero `src/` references / no writer found for: `dspy_optimization_runs`,
`dspy_prompt_versions`, `dspy_cognitive_context_history` (migration `014`)
and `investigation_hops` (`001`, FK child of the orphaned
`cognitive_cycles`). These are empty, harmless, but add schema surface and
reader confusion.

> Note: `dspy_agent_training_signals` (same migration) is **not** orphaned —
> it has a real writer (`rag/memory_adapters.py:779` `flush → insert`) and
> reader (`get_signals_for_optimization`, `:814`), fed by per-agent
> `SignalCollector`s. (One sub-audit wrongly grouped it with the orphans.)

**Classification:** DELETE-candidate **pending intent check** — migration
`014` may be a deliberate stake for in-flight DSPy optimization. Confirm
against the DSPy roadmap before dropping.

### F4 — Inert cache controls  · LOW

`falkordb_synced` / `falkordb_sync_at` (`001`:334) are never updated and
never read — there is no Supabase→FalkorDB sync-back job. `semantic_cache_
ttl_minutes` (config `:397`) is loaded into `GraphitiConfig.cache_ttl_
minutes` but no code enforces eviction (Postgres has no auto-TTL). Both are
no-ops today; only relevant if F2's cache is activated. **Classification:**
implement-or-remove alongside the F2 decision.

### F5 — Invalidation cascade scope  · INFO / by-design

`INVALIDATABLE_TABLES` = `{triggers, ml_predictions, executive_insights}`;
`procedural_memories` and `semantic_memory_cache` are deliberately excluded.
**Reasoning (not a bug):** procedural memories are *generalized, reusable
how-to patterns* gated behind usage/success thresholds — not dataset-bound
findings — so an overturned causal result need not invalidate them; the
cache is derived from the data layer, not from invalidatable insights. One
sub-audit flagged this "HARMFUL-NOW," but that's rule-matching ("cascade
should be comprehensive") without reasoning about what procedural memory
*represents*. **Recommendation:** leave as-is; add a one-line code comment
recording the deliberate exclusion, and confirm with the subsystem owner.

### F6 — Documentation clarity  · LOW

`ARCHITECTURE.md` / `07-SUPPORTING-SCHEMAS.md` present `cognitive_cycles` as
the 4-phase cognitive store without noting that (a) it is currently written
by nothing and (b) live conversation history lives in
`chatbot_conversations`. A short "status: scaffolded, not in the live path;
conversation history is served by `chatbot_conversations`" note would
prevent the next reader from trusting it.

---

## 5. Claims investigated and DISPROVEN

Recorded for transparency — these were asserted by the read-only
sub-audits and **rejected** after verification. They are the strongest
evidence that the user's "do not pattern-match" instruction was the right
call.

| # | Claim | Verdict | Disproving evidence |
|---|-------|---------|---------------------|
| 5.1 | "Memory tables: 7 claimed vs 14 actual → doc drift" | **FALSE** | `001` core schema creates exactly 7 memory tables |
| 5.2 | "`dspy_agent_training_signals` orphaned, no writer" | **FALSE** | `rag/memory_adapters.py:779` inserts, `:814` reads |
| 5.3 | "`update_procedure_outcome` never called / counters write-only" | **FALSE** | called at `feedback_learner/memory_hooks.py:193` and `routes/memory.py:662` |
| 5.4 | "`success_rate` is a dead field / not a column" | **FALSE** | DB `GENERATED` column `001`:261; used in `find_similar_procedures` ranking `001`:707 |
| 5.5 | "HPO warm-start never feeds Optuna" | **FALSE** | `study.enqueue_trial(...)` `hyperparameter_tuner.py:365` + `record_warmstart_outcome` |
| 5.6 | "Crystallizer deterministic prose is a HARMFUL-NOW silent mock" | **FALSE** (see below) | derived from real `cohort_size`/`sensitivity_checks_failed`/`effect_direction`; generic branch self-labels `"(deterministic heuristic)"` |
| 5.7 | "`hybrid_vector_search`/`hybrid_fulltext_search` RPCs unwired" | **FALSE** | called by `rag/memory_connector.py:144,248`; `backends/vector.py:29` |
| 5.8 | "`cognitive_cycles` is FULLY WIRED" | **FALSE** | no `src/` writer; see F1 — it is orphaned/superseded |

#### 5.6 detail — "Crystallizer deterministic prose is a harmful silent mock" → FALSE

This was the only candidate for a plan-354-style harmful fake, so it got
extra scrutiny. `_deterministic_narrative_prose`
(`crystallizer.py:786`), used when the LLM narrator flag is off, builds
`limitations` from **real derived values** (`cohort_size`,
`sensitivity_checks_failed`, `effect_direction`) and its only generic branch
emits the literal string **`"standard limitations apply (deterministic
heuristic)"`** — i.e. it **self-labels** as a heuristic. That is the
*opposite* of a silent mock returning plausible-wrong numbers; it is honest,
data-derived, and labeled. A minor enhancement would be to surface
LLM-vs-heuristic provenance at the API boundary (today it's only inferable
from the presence/absence of a `crystal_narrative_audits` row), but this is
a LOW transparency nicety, **not** a harmful finding.

---

## 6. Recommendations (in priority order)

1. **Decide the fate of the `cognitive_cycles` trio (F1).** Recommended:
   retire `conversation.py` + the `016` RPCs (superseded by
   `chatbot_conversations`); keep or drop the `cognitive_cycles` /
   `investigation_hops` tables per owner confirmation. If kept for a future
   feature, fix the RPC's 7 phantom columns so it isn't a landmine.
2. **Resolve `semantic_memory_cache` (F2/F4).** Either schedule the sync +
   wire a reader (and add TTL eviction + `falkordb_synced` sync-back), or
   document it as deploy-seed-only and remove the inert controls.
3. **Triage the orphan tables (F3)** against the DSPy roadmap; drop the
   three truly-unreferenced `dspy_*` tables if not staked.
4. **Add by-design comments** for the invalidation-scope exclusion (F5) and
   a status note distinguishing `cognitive_cycles` from
   `chatbot_conversations` (F6).
5. **Add a CI guard that diffs RPC column references against table DDL** —
   the `016` defect is exactly the class of bug mocked unit tests miss; a
   cheap static check (like the one in this audit's appendix) would have
   caught it at commit time.

No code was changed by this audit. None of the findings warrant a
unilateral delete; each carries recoverable intent that should be confirmed
with the subsystem owner first (per REASON-BEFORE-RULES).

---

## 7. Appendix — producer / consumer map (verified)

| Table | Producer (write) | Consumer (read) | Status |
|-------|------------------|-----------------|--------|
| `episodic_memories` | 22 agent memory-hooks; reflector | consolidator; HybridRetriever (vector/FT RPCs); crystallizer | **WIRED** |
| `procedural_memories` | tool_composer / resource_optimizer hooks; cognitive_rag; consolidator; `update_procedure_outcome` | `find_similar_procedures` (ranks by `success_rate`); few-shot | **WIRED** |
| `executive_insights` | crystallizer | sentinels; API routes; invalidator | **WIRED** |
| `insight_edges` | consolidator / crystallizer | invalidator BFS | **WIRED** |
| `learning_signals` | reflector `record_learning_signal` | memory routes / observability | **WIRED** |
| `dspy_agent_training_signals` | `memory_adapters` flush (per-agent collectors) | `get_signals_for_optimization` | **WIRED** |
| `semantic_memory_cache` | sync wrappers — **never called** | **none in live code** | **DORMANT (F2)** |
| `cognitive_cycles` | **none** | `ConversationRepository` (never instantiated) via **broken** `016` RPC | **ORPHANED (F1)** |
| `investigation_hops` | **none** | **none** | **ORPHANED (F3)** |
| `dspy_optimization_runs` / `dspy_prompt_versions` / `dspy_cognitive_context_history` | **none** | **none** | **ORPHANED (F3)** |
| `memory_statistics` | (not deeply verified) | observability | not verified |

## 8. Appendix — verification log (commands executed)

- Enumerate core memory tables → `grep "CREATE TABLE" 001_agentic_memory_schema_v1.3.sql` → 7 tables incl. `semantic_memory_cache`.
- `cognitive_cycles` writers → `grep -E '\.table\("cognitive_cycles"\)\.(insert|upsert|update)' src/` → **none**.
- `ConversationRepository` instantiation → `grep -rn "ConversationRepository" src/` → export only.
- **016 RPC column cross-check** → Python parse of the `cognitive_cycles`
  DDL vs `cc.<col>` references → 7 phantom columns; verdict BROKEN.
- Sync wrapper callers → `grep` for `sync_data_layer_to_semantic_cache` /
  `sync_treatment_relationships_to_cache` → **zero callers**; beat schedule
  has no semantic-cache entry.
- Hybrid RPC callers → `grep "hybrid_vector_search|hybrid_fulltext_search" src/`
  → `memory_connector.py:144,248`, `backends/vector.py:29` (WIRED).
- Procedure outcome callers → `feedback_learner/memory_hooks.py:193`,
  `routes/memory.py:662`.
- `success_rate` → `GENERATED` column `001`:261; ranking use `001`:707.
- HPO warm-start → `study.enqueue_trial` `hyperparameter_tuner.py:365`.
- Live-ness sanity → `episodic_memory` in 22 agent files, `procedural_memory`
  in 7.

*Verification performed at source/SQL-schema level (the faithful environment
for wiring/orphan/drift questions). No live Postgres/Redis/FalkorDB or
`pytest` run was available in this container.*
