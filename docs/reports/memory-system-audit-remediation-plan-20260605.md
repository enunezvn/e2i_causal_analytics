# Memory System Audit (2026-06-05) — Sharded Remediation Plan

**Source audit:** `docs/reports/memory-system-audit-20260605.md` (audit branch `claude/bold-galileo-DULyN`)
**Plan date:** 2026-06-05
**Scope:** Resolve all 6 documented findings (F1–F6) + the recommended CI guard, on the memory subsystem (`episodic_memories`, `procedural_memories`, `semantic_memory_cache`, `cognitive_cycles`, `investigation_hops`, `learning_signals`, `memory_statistics`, the lifecycle, crystallization, sentinels, triple-stream retrieval, DSPy peripherals).

> **Governing principle (from the audit + repo CLAUDE.md):** *"None of the findings warrant a unilateral delete; each carries recoverable intent that should be confirmed with the subsystem owner first."* Every retire/drop shard is therefore **gated on a Phase-0 decision**. Do not skip Phase 0.

---

## 0. How to use this plan (context-window-friendly execution model)

This plan is **sharded**: each shard is a self-contained unit sized to run in **one fresh context window** (one agent / one worktree), with its own *Cold-start context* (the only files to read), objective, change set, red-first test, and acceptance bar. You do **not** need the whole audit or this whole file loaded to execute a shard — load only the shard block + its Cold-start list.

**Per-shard methodology (the repo's standing convention, from issue #694):**
`isolated git worktree` → **TDD red-first** (write the failing test/repro that asserts the target state) → drive to green → `ruff check` + `mypy --config-file pyproject.toml src/` clean → `codex:codex-rescue` to fixed-point **ACCEPT** → PR (draft) → (for DB/migration shards) **faithful-environment verify** (`docker exec` into the droplet, not the host `.venv`) → mark done with recorded evidence.

**Sizing:** each shard ≈ 1–4 files touched + 1 test file. If a shard grows past that, split it.

**Branch/worktree naming:** `claude/mem-<shardid>-<slug>` (e.g. `claude/mem-S2.1-retire-conversation`).

---

## 1. Findings → shards map

| Finding | Sev | Decision gate | Execution shards |
|---|---|---|---|
| **F1** `cognitive_cycles` trio orphaned; `016` RPC has 7 phantom columns | MEDIUM | **D1** | S2.1, S2.2 (RETIRE) **or** S2.1′ (REWIRE) |
| **F2** `semantic_memory_cache` dormant both ends | LOW–MED | **D2** | S3.1–S3.3 (ACTIVATE) **or** S3.1′ (RETIRE) |
| **F4** inert cache controls (`falkordb_synced`, TTL) | LOW | **D2** (coupled) | folded into S3.x |
| **F3** orphan tables (`dspy_optimization_runs`, `dspy_prompt_versions`, `dspy_cognitive_context_history`, `investigation_hops`) | LOW | **D3** | S4.1 (DROP) **or** S4.1′ (KEEP+doc) |
| **F5** invalidation cascade scope (by-design exclusion) | INFO | none | **S1.1** (comment only) |
| **F6** docs don't flag orphaned `cognitive_cycles` | LOW | none | **S1.2** |
| **Rec 5** CI guard: RPC column refs vs table DDL | (new) | none | **S1.3** |

---

## 2. Dependency graph & recommended order

```
Phase 0  D1 ─┐   D2 ─┐   D3 ─┐          (decisions; no code)
             │       │       │
Phase 1  S1.1  S1.2  S1.3   ← safe, additive, NO decision needed → DO FIRST (parallel)
             │       │       │
Phase 2  D1 → { S2.1 → S2.2 }  (RETIRE)  |  S2.1′ (REWIRE)     ← F1, highest value
             │
Phase 3  D2 → { S3.1 ∥ S3.2 → S3.3 } (ACTIVATE) | S3.1′ (RETIRE)  ← F2/F4
             │
Phase 4  D3 → S4.1 (DROP) | S4.1′ (KEEP+doc)   ← F3
```

- **Phase 1 is unblocked** — ship it immediately to lock value while decisions are pending.
- **S1.3 (CI guard)** ships in *report-only* mode in Phase 1; it is **flipped to blocking by the F1 shard** (Phase 2) once the `016` defect is resolved (otherwise it would red-fail CI on a known baseline). See S1.3 + S2.1.
- Phases 2/3/4 are mutually independent once their decision lands; run in parallel if you have the worktrees.

---

# Phase 0 — Decision gates (no code; unblocks the retire/drop shards)

Each gate has a **recommended default** (the audit's own recommendation) and the **evidence** to confirm it. Record the answers in a short decision log (`docs/plans/memory-remediation-decisions-20260605.md`) before starting the gated shards.

### D1 — Fate of the `cognitive_cycles` trio (F1)
- **Question:** Retire the superseded trio, or rewire conversation-history onto `cognitive_cycles`?
- **Audit recommendation (default):** **RETIRE.** `ConversationRepository` is export-only/never instantiated; the live store is `chatbot_conversations` (`copilotkit.py:945`); the 4-phase workflow persists to `episodic_memories`+`learning_signals`+FalkorDB, never to `cognitive_cycles`. The `016` RPC references 7 columns absent from the DDL.
- **Sub-decision (D1b):** after retiring the repo+RPC, also **DROP** the `cognitive_cycles` + `investigation_hops` *tables*? (Default: yes, if no roadmap stake.)
- **Evidence to reconfirm before acting:** `grep -rn "ConversationRepository(" src/` returns 0 instantiations; `grep -rnE 'table\("cognitive_cycles"\)\.(insert|upsert|update)' src/` returns 0.
- **→ RETIRE selects S2.1 + S2.2. REWIRE selects S2.1′.**

### D2 — `semantic_memory_cache` (F2 + F4)
- **Question:** Activate the Supabase hot-cache mirror, or retire the scaffolding?
- **Audit recommendation (default):** **decide against the cache roadmap.** If the cache is roadmapped → ACTIVATE (sync beat + reader + TTL eviction + `falkordb_synced` sync-back). If not → RETIRE wrappers + remove inert controls + document the table as deploy-seed-only.
- **Evidence:** sync wrappers (`episodic_memory.py:884`, `semantic_memory.py:1579`) have 0 `src/` callers; no live reader of `semantic_memory_cache`; no Celery-beat entry.
- **→ ACTIVATE selects S3.1–S3.3. RETIRE selects S3.1′.**

### D3 — Orphan DSPy tables (F3)
- **Question:** Drop the three unreferenced `dspy_*` tables (and `investigation_hops` if not already handled by D1b), or keep as a DSPy-roadmap stake?
- **Audit recommendation (default):** **DELETE-candidate pending DSPy-roadmap check.** Note `dspy_agent_training_signals` is **live** (writer `rag/memory_adapters.py:779`, reader `:814`) — **do not** touch it.
- **Evidence:** `grep -rn "dspy_optimization_runs\|dspy_prompt_versions\|dspy_cognitive_context_history" src/` returns 0.
- **→ DROP selects S4.1. KEEP selects S4.1′.**

---

# Phase 1 — Safe, additive shards (no decision; do first, in parallel)

### Shard S1.1 — F5: document the by-design invalidation-cascade exclusion
- **Finding:** F5 (INFO/by-design). `INVALIDATABLE_TABLES = {triggers, ml_predictions, executive_insights}` deliberately excludes `procedural_memories` & `semantic_memory_cache`.
- **Cold-start context:** `src/memory/invalidator.py` (the `INVALIDATABLE_TABLES` definition); audit §F5.
- **Objective:** make the deliberate exclusion explicit so a future reader doesn't "fix" it.
- **Changes:** add a one-line+rationale comment above `INVALIDATABLE_TABLES` (procedural = generalized reusable how-to, not dataset-bound; cache = derived from data layer, not from invalidatable insights).
- **Red-first test:** a characterization test asserting `procedural_memories` and `semantic_memory_cache` are **NOT** in `INVALIDATABLE_TABLES` (locks the intent; fails if someone "comprehensively" adds them).
- **Acceptance:** test green; ruff/mypy clean; codex ACCEPT. No deploy (pure code+comment).

### Shard S1.2 — F6: doc status note (orphaned `cognitive_cycles` vs live `chatbot_conversations`)
- **Finding:** F6 (LOW, doc clarity).
- **Cold-start context:** `docs/ARCHITECTURE.md` (§ cognitive store), `docs/data/07-SUPPORTING-SCHEMAS.md` (the `cognitive_cycles` entry).
- **Objective:** add a status callout: *"`cognitive_cycles` is scaffolded, NOT in the live path; conversation history is served by `chatbot_conversations`. The 4-phase cognitive workflow persists to `episodic_memories`+`learning_signals`+FalkorDB."*
- **Changes:** the two doc files only.
- **Red-first test:** N/A (docs) — instead, a `grep` assertion in the PR description that both files now contain the status string. Optionally add to a docs-lint if one exists.
- **Acceptance:** reviewer confirms; no deploy. *Sequencing:* if D1=RETIRE, this note is superseded by S2.x's removal — keep S1.2 minimal or fold into S2.2's doc update. Safe to ship now regardless.

### Shard S1.3 — Rec 5: CI guard diffing RPC column references vs table DDL
- **Finding:** Recommendation 5 (the static check that would have caught the `016` defect).
- **Cold-start context:** `database/memory/016_conversation_similarity_search.sql` (the broken RPC, as the guard's first test fixture), `database/memory/001_agentic_memory_schema_v1.3.sql` (the `cognitive_cycles` DDL), `.github/workflows/feature_contract_guard.yml` + `g3_wiring_guard.yml` (convention to mirror), audit §8 appendix (the exact cross-check method).
- **Objective:** a script that parses every `CREATE FUNCTION` referencing `<alias>.<col>` against the referenced table's DDL and reports phantom columns; wired into CI.
- **Changes:**
  1. `scripts/ci/rpc_ddl_column_guard.py` — parse `database/**/*.sql`; for each RPC, resolve table aliases → DDL columns → flag references to non-existent columns. (Port the audit's appendix cross-check.)
  2. `.github/workflows/rpc_ddl_guard.yml` — runs the script on PRs touching `database/**`.
- **Red-first test:** `tests/ci/test_rpc_ddl_column_guard.py` asserting the guard **flags the `016` RPC's 7 phantom columns** (`agent_response, created_at, feedback_at, feedback_score, feedback_text, feedback_type, response_type`) — proving it catches the real defect — and passes on a known-clean RPC (e.g. `find_similar_procedures`).
- **Acceptance / sequencing:** ship the guard in **report-only / non-blocking** mode with the `016` finding recorded as a **known baseline** (the guard prints it but exits 0), because `016` is still on `main` until F1 lands. **The F1 RETIRE/REWIRE shard (S2.1) removes the `016` defect AND flips this guard to blocking + deletes the baseline.** ruff/mypy/codex ACCEPT.

---

# Phase 2 — F1 resolution (MEDIUM) — gated on D1

## If D1 = RETIRE (recommended)

### Shard S2.1 — retire `ConversationRepository` + the `016` RPCs
- **Cold-start context:** `src/repositories/conversation.py`, `src/repositories/__init__.py` (exports at lines ~75,181), `database/memory/016_conversation_similarity_search.sql`, the S1.3 guard.
- **Discovery step (do first):** `grep -nE "CREATE (OR REPLACE )?FUNCTION" database/memory/016_conversation_similarity_search.sql` → record exact RPC function name(s)+signatures to drop.
- **Changes:**
  1. Delete `src/repositories/conversation.py`; remove its export + `__all__` entry from `repositories/__init__.py`.
  2. New forward migration `database/memory/0XX_retire_conversation_similarity_rpcs.sql` — `DROP FUNCTION IF EXISTS <each 016 RPC>(...) CASCADE;` idempotent; tracked in `schema_migrations`.
  3. Flip the S1.3 guard to **blocking** and remove the `016` baseline entry.
- **Red-first test:** (a) a test asserting `from src.repositories import ConversationRepository` raises `ImportError` (symbol removed); (b) an **anti-resurrection guard** test (mirror #698 pattern) asserting no `src/` reference to the dropped RPC names; (c) the S1.3 guard now exits non-zero if `016`'s RPCs reappear.
- **Acceptance:** tests green; ruff/mypy/codex ACCEPT; **faithful verify** the migration runs idempotently on the droplet (`docker exec supabase-db`) and the functions are gone.

### Shard S2.2 — drop `cognitive_cycles` + `investigation_hops` tables (only if D1b = DROP)
- **Depends on:** S2.1 (RPCs referencing the table must be gone first).
- **Cold-start context:** `database/memory/001_agentic_memory_schema_v1.3.sql` (table + FK `investigation_hops → cognitive_cycles`), audit Appendix §7.
- **Changes:** forward migration `database/memory/0XX_drop_cognitive_cycles_trio.sql` — drop `investigation_hops` (FK child) then `cognitive_cycles`, `IF EXISTS`, idempotent, tracked. Update `docs/data/07-SUPPORTING-SCHEMAS.md` to remove/annotate the entries (supersedes S1.2's note).
- **Red-first test:** anti-resurrection guard asserting neither table is re-declared in any non-archived migration; migration re-run is idempotent.
- **Acceptance:** faithful verify (tables absent on droplet); codex ACCEPT. *Note:* this also resolves the `investigation_hops` half of F3 — coordinate with D3/S4.1 so it isn't double-dropped.

## If D1 = REWIRE (alternative)

### Shard S2.1′ — fix the `016` RPC columns + wire a producer
- **Cold-start context:** `016_conversation_similarity_search.sql`, the `cognitive_cycles` DDL in `001`, `src/repositories/conversation.py`.
- **Changes:** (1) new migration rewriting the RPC to reference real columns (`synthesized_response`, `confidence_score`, `started_at`/`completed_at`; replace the 7 phantom `feedback_*`/`agent_response`/`created_at`/`response_type` refs per the intended semantics — add the missing columns via migration **only if** the feedback feature is actually wanted); (2) wire a real producer that writes `cognitive_cycles`; (3) instantiate `ConversationRepository` where intended.
- **Red-first test:** a test that executes the RPC against a seeded `cognitive_cycles` row and returns a similarity match (catches the schema drift the mock tests missed); producer round-trip test.
- **Acceptance:** faithful verify on droplet; S1.3 guard green + blocking; codex ACCEPT. *(Higher effort than RETIRE; only if owner wants the feature.)*

---

# Phase 3 — F2 + F4 resolution — gated on D2

## If D2 = RETIRE (default unless cache is roadmapped)

### Shard S3.1′ — retire dormant sync wrappers + inert controls + document
- **Cold-start context:** `src/memory/episodic_memory.py:884` (`sync_treatment_relationships_to_cache`), `src/memory/semantic_memory.py:1579` (`sync_data_layer_to_semantic_cache`), the `falkordb_synced`/`falkordb_sync_at` columns (`001`:334), `semantic_cache_ttl_minutes` config (`:397` → `GraphitiConfig.cache_ttl_minutes`), `001b` populating RPC.
- **Changes:** (1) remove the two never-called sync wrappers (or, if the RPC `sync_hcp_patient_relationships_to_cache` is deploy-seed useful, keep the RPC and remove only the dead Python wrappers); (2) remove the inert `cache_ttl_minutes` plumbing and the unused `falkordb_synced` read paths (F4); (3) add a header comment to the `semantic_memory_cache` DDL / `07-SUPPORTING-SCHEMAS.md`: *"deploy-seed-only; no live producer/reader (audit 2026-06-05 F2)."*
- **Red-first test:** anti-resurrection guard asserting zero callers re-appear; characterization test that the cache table is documented as seed-only (or simply that the removed symbols are gone).
- **Acceptance:** ruff/mypy/codex ACCEPT. No deploy if pure-Python; if a migration touches the table comment, faithful verify.

## If D2 = ACTIVATE (only if the cache is on the roadmap)

### Shard S3.1 — schedule the sync (Celery beat)
- **Cold-start context:** the Celery beat schedule module, the two sync wrappers.
- **Changes:** add a beat entry invoking the sync on a cadence; idempotency/lock guard.
- **Red-first test:** beat-registry test asserting the task is scheduled; a faithful run persists rows to `semantic_memory_cache`.

### Shard S3.2 — wire a reader (graph path consumes the cache)
- **Cold-start context:** `src/rag/memory_connector.py`, `HybridRetriever` backends.
- **Changes:** route the relationship lookups through `semantic_memory_cache` (Supabase hot path) with FalkorDB fallback.
- **Red-first test:** retrieval test that a cache hit short-circuits the graph call.

### Shard S3.3 — TTL eviction + `falkordb_synced` sync-back (F4)
- **Depends on:** S3.1.
- **Changes:** a TTL-eviction job (Postgres has no auto-TTL) honoring `semantic_cache_ttl_minutes`; a sync-back that stamps `falkordb_synced`/`falkordb_sync_at`.
- **Red-first test:** rows past TTL are evicted; `falkordb_synced` is set after sync-back.
- **Acceptance (all S3.x):** faithful verify on droplet; codex ACCEPT.

---

# Phase 4 — F3 resolution — gated on D3

## If D3 = DROP (default, pending DSPy-roadmap check)

### Shard S4.1 — drop the three orphan `dspy_*` tables (+ `investigation_hops` if not done in S2.2)
- **Cold-start context:** the migration that creates `dspy_optimization_runs`/`dspy_prompt_versions`/`dspy_cognitive_context_history` (discover: `grep -rln "CREATE TABLE.*dspy_optimization_runs" database/`), audit §F3 (and its **do-not-touch** note on `dspy_agent_training_signals`).
- **Changes:** forward migration `database/0XX_drop_orphan_dspy_tables.sql` — `DROP TABLE IF EXISTS` the three orphans (and `investigation_hops` only if S2.2 didn't); idempotent; tracked. **Explicitly exclude `dspy_agent_training_signals`.**
- **Red-first test:** anti-resurrection guard for the three table names; assert `dspy_agent_training_signals` still exists and is still written by `memory_adapters.py:779` (guard against accidental over-drop).
- **Acceptance:** faithful verify (orphans gone, live table intact); codex ACCEPT.

## If D3 = KEEP

### Shard S4.1′ — document the DSPy stake
- **Changes:** add a "DSPy-optimization roadmap stake; intentionally empty (audit 2026-06-05 F3)" note to the migration header + `07-SUPPORTING-SCHEMAS.md`; optionally a test asserting they remain empty/unreferenced so they don't silently grow load-bearing.

---

## Appendix A — per-shard checklist (paste into each shard's PR)

```
[ ] worktree:  git worktree add ../wt-<shardid> -b claude/mem-<shardid>-<slug>
[ ] red-first: failing test asserting target state committed FIRST
[ ] green:     change drives the test to pass
[ ] ruff:      ruff check src/ tests/   (clean)
[ ] mypy:      mypy --config-file pyproject.toml src/   (no new errors)
[ ] codex:     codex:codex-rescue → fixed-point ACCEPT
[ ] DB only:   faithful verify on droplet (docker exec supabase-db), idempotent re-run, schema_migrations tracked
[ ] anti-resurrection guard added for any deletion
[ ] PR (draft) opened; decision-log link if gated
```

## Appendix B — verification one-liners (reconfirm before acting)

```bash
# D1
grep -rn "ConversationRepository(" src/ | grep -v def          # expect 0 instantiations
grep -rnE 'table\("cognitive_cycles"\)\.(insert|upsert|update)' src/   # expect 0 writers
grep -nE "CREATE (OR REPLACE )?FUNCTION" database/memory/016_conversation_similarity_search.sql
# D2
grep -rn "sync_data_layer_to_semantic_cache\|sync_treatment_relationships_to_cache" src/   # callers? expect 0
# D3
grep -rn "dspy_optimization_runs\|dspy_prompt_versions\|dspy_cognitive_context_history" src/   # expect 0
grep -rn "dspy_agent_training_signals" src/   # expect writer :779 + reader :814 (DO NOT DROP)
```

## Appendix C — what NOT to touch (audit §2, §5 — verified-correct hot path)

Episodic capture/dedup (`consolidator._compute_dedup_signature` + migration `026` index); config-driven promotion thresholds; `procedural_memories` wiring (`update_procedure_outcome`, `success_rate` GENERATED column, `find_similar_procedures` ranking); crystallizer (deterministic prose is honest/self-labeled — **not** a silent mock, §5.6); cascade invalidation BFS; triple-stream retrieval RPCs; HPO warm-start; `dspy_agent_training_signals`. **Leave these alone.**
