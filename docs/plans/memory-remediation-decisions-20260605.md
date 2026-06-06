# Memory-System Remediation — Decision Log (D1 / D2 / D3)

> **Status: retroactive record.** The `.claude/plans/memory-system-remediation`
> plan required a Phase-0 decision log before the gated retire/drop shards. The
> shards were executed (PR **#732** / commit `a027643a` + commit `9cb0dc19`) and
> deployed before this file was written; this document records the gates that
> were actually resolved, with the functional-liveness evidence and the
> intent reasoning, so the artifact the plan asked for exists.
>
> Source audit: `docs/reports/memory-system-audit-20260605.md`.
> Governing discipline: **REASON-BEFORE-RULES** (CLAUDE.md) — investigate intent
> (PR/issue/git history) → understand the shape → assess present harm → classify.
> Owner directive applied to every gate: **"do not omit features fished for
> later."** A *superseded* design (replaced by a live equivalent) is vestigial
> and safe to drop; a *staked-for-later* feature is not. Each decision below
> shows which of those it is, with evidence.

---

## D1 — fate of the `cognitive_cycles` trio (F1) → **RETIRE + DROP**

**Decision:** Retire `ConversationRepository` + the `016` similarity RPCs
(migration `031`), and drop the `cognitive_cycles` + `investigation_hops` tables
(migration `032`, D1b = DROP).

**Evidence (data-driven, reconfirmed 2026-06-06):**
- **0 writers** in `src/`: `grep -E 'table\("cognitive_cycles"\)\.(insert|upsert|update)' src/` → none.
- `ConversationRepository` is **never instantiated** — the only live conversation
  repository is `ChatbotConversationRepository` (`src/api/routes/copilotkit.py:945`).
- The `016` RPC referenced **7 columns absent from the DDL** (`agent_response,
  created_at, feedback_at, feedback_score, feedback_text, feedback_type,
  response_type`) → it would raise `UndefinedColumn` if ever executed. **Broken
  since inception; never wired.**
- Timeline: `conversation.py` + `016` were added in the initial platform commit
  (`3e1c70cf`, 2025-12-20); `chatbot_conversations` arrived ~3 weeks later with
  the CopilotKit chatbot feature (`a1119f2a`, 2026-01-08) and became the live
  conversation store. The live 4-phase cognitive workflow persists to
  `episodic_memories` + `learning_signals` + FalkorDB — **never** to
  `cognitive_cycles`.

**Intent classification: SUPERSEDED / vestigial — not fished-for-later.** The
conversation-history feature is **preserved** (served by `chatbot_conversations`),
and the cognitive-workflow feature is **preserved** (served by episodic +
learning_signals + FalkorDB). What was dropped is a never-functional, replaced
alternative store. No requested-for-later capability was lost. DROP is defensible.

**Faithful verification (droplet `supabase-db`):** `cognitive_cycles` +
`investigation_hops` **absent**; the `cycle_id` columns on `episodic_memories`
and `learning_signals` **remain** (the `DROP ... CASCADE` removed the FK
constraints but kept the columns — now plain orphaned UUIDs, documented as such
in `07-SUPPORTING-SCHEMAS.md`).

---

## D2 — `semantic_memory_cache` (F2 + F4) → **RETIRE dead wrappers, KEEP the substrate**

**Decision:** Remove the two never-called sync wrappers
(`sync_data_layer_to_semantic_cache`, `sync_treatment_relationships_to_cache`)
and the inert `semantic_cache_ttl_minutes` config control (commit `9cb0dc19`).
The inert `falkordb_synced` / `falkordb_sync_at` columns were **deferred** by
`9cb0dc19` and dropped later by migration `034_drop_inert_falkordb_sync_columns.sql`
(audit-followup 2026-06-06); the stray `semantic_cache_ttl_minutes` key was also
removed from `config/005_memory_config.yaml` then. **Keep** the
`semantic_memory_cache` table and its populating RPC
(`sync_hcp_patient_relationships_to_cache`).

**Evidence:**
- `grep "sync_data_layer_to_semantic_cache\|sync_treatment_relationships_to_cache" src/`
  → **0 callers**, no Celery-beat entry; no live reader of the table.
- The table + populating RPC are **real and correct** (audit §F2) — the only
  defect was that nothing connected either end.

**Intent classification: STAKED-FOR-LATER substrate + dead glue code.** This is
the case the owner directive most directly protects. The resolution removed only
**zero-caller dead Python wrappers + no-op config controls**, while
**preserving the data substrate** (table + RPC) as scaffolding for a future
FalkorDB→Supabase hot-cache mirror. Activation later requires only a sync job +
a reader + TTL eviction — nothing was foreclosed. The "fished-for-later" feature
is intact; its dead glue was removed. (Documented inline in
`07-SUPPORTING-SCHEMAS.md` as "deploy-seed-only".)

**Faithful verification:** `semantic_memory_cache` table **present** in the
droplet DB.

---

## D3 — orphan `dspy_*` tables (F3) → **DROP 3 orphans, PRESERVE the live DSPy substrate**

**Decision:** Drop `dspy_optimization_runs`, `dspy_prompt_versions`,
`dspy_cognitive_context_history` (migration `033`). **Do not touch**
`dspy_agent_training_signals` or the newer GEPA `023` tables.

**Evidence:**
- `grep "dspy_optimization_runs\|dspy_prompt_versions\|dspy_cognitive_context_history" src/`
  → **0 references**; no writer/reader.
- `dspy_agent_training_signals` is **LIVE**: writer `src/rag/memory_adapters.py:779`
  (`insert`), reader `:814` (`get_signals_for_optimization`).
- The three dropped tables are the **MIPROv2-era** optimization scaffolding,
  superseded by the GEPA approach (migration `023`).

**Intent classification: SUPERSEDED optimization design — not fished-for-later.**
The DSPy-optimization *feature* is preserved: the live signal table + the GEPA
`023` tables carry it. The dropped tables are an older, replaced design with no
references. DROP defensible; the live substrate was explicitly excluded from the
drop (and a regression guard asserts `dspy_agent_training_signals` survives).

**Faithful verification:** the three orphans **absent**;
`dspy_agent_training_signals` **present** in the droplet DB.

---

## Other findings (no gate)

- **F5** (invalidator exclusion) — documented rationale comment +
  characterization test (`a027643a`). The deliberate exclusion of
  `procedural_memories` / `semantic_memory_cache` from `INVALIDATABLE_TABLES` is
  by design (procedural = generalized reusable patterns, not dataset-bound;
  cache = derived, not an invalidatable insight).
- **Rec 5** (RPC↔DDL column guard) — `scripts/ci/rpc_ddl_column_guard.py` +
  `.github/workflows/rpc_ddl_guard.yml`, flipped to **blocking** (`a027643a`).
- **F6** (docs clarity) — closed by **this** PR: `07-SUPPORTING-SCHEMAS.md` +
  `ONBOARDING.md` now reflect the DROP reality (RETIRED markers + migration
  provenance; `semantic_memory_cache` flagged seed-only), with a red-first
  anti-resurrection guard test.

---

## Summary — was any "feature fished for later" omitted?

**No.** Every drop removed a **superseded** design whose user-facing capability
is served by a **live equivalent** (conversation history → `chatbot_conversations`;
cognitive workflow → episodic/learning_signals/FalkorDB; DSPy optimization →
`dspy_agent_training_signals` + GEPA `023`). The one genuinely staked-for-later
piece — the `semantic_memory_cache` hot-cache mirror — had its **substrate
preserved**; only zero-caller dead glue was removed. If the owner nonetheless
wants any retired artifact restored, it is recoverable from git history +
the original migrations; flag it and a forward restore migration can be written.
