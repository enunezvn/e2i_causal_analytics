# E2I Memory Subsystem — Deep Code Review

**Date:** 2026-06-03
**Reviewer:** Multi-agent workflow (57 agents) + dispatcher independent verification
**Type:** Read-only review — the review below was read-only; the **Update** section that follows records the follow-up session's faithful (prod-droplet) verification + remediation, which **corrects several findings** (notably H3).

---

## ⚡ UPDATE — 2026-06-03 follow-up session (faithful verification, remediation, reconciliation plan)

> The review below was performed read-only against `main`. This session verified findings in the **faithful environment** (the production droplet, via `docker exec` into `e2i_api` / `supabase-db` — **not** the host `.venv`), which corrected conclusions, and applied a large backlog of never-deployed migrations.

### Faithful-environment corrections
- **H3 (sync RedisSaver) → effectively a NON-ISSUE (LOW), superseding the HIGH below.** Inside `e2i_api`: prod Redis is plain `redis:7-alpine` (no RediSearch), so `RedisSaver.setup()` fails (`unknown command 'FT._LIST'`) and `get_langgraph_checkpointer()` silently falls back to `InMemorySaver` — the sync-saver `NotImplementedError` path is never even reached. AND conversation continuity comes from **Supabase chat repos** (`chatbot_graph.load_context_node` reloads history each turn), so the LangGraph checkpointer is **vestigial**, not load-bearing. **Do NOT apply the AsyncRedisSaver "fix"** — it also requires RediSearch.
- **NEW (high-impact): the prod embedding service is DOWN.** `get_embedding_service()` → `FallbackEmbeddingService` raises `All embedding services unavailable`: no `OPENAI_API_KEY` is forwarded by compose (primary `OpenAIEmbeddingService` can't init), and the read-only container blocks the local fallback model. Every episodic write needs an embedding → fails → `episodic_memories` = **0 rows** → the consolidation/crystallization lifecycle has no input. **Fix queued in PR #672** (forward `OPENAI_API_KEY` via `x-common-env`); needs a container recreate/deploy to take effect.
- **The memory lifecycle was NON-OPERATIONAL at the schema level.** The entire `database/memory/` set (014→029) — `executive_insights`, `insight_edges`, the `sentinels` table, `invalidated_at` columns, episodic `dedup_*` — was never applied to the droplet. **Root cause (2-part):** `deploy.yml` gated migrations on `SUPABASE_DB_URL` (unset on the droplet, REST-creds-only) → always skipped; AND `run_migrations.sh` only scanned `database/migrations/`, never `database/memory/`.

### Staleness model — intended vs reality
- **Intended (ARCHITECTURE.md §5):** Redis = short-term (24h-TTL working memory); Supabase + FalkorDB = long-term. Staleness disambiguated via binary `invalidated_at` (NULL = fresh) + `cascade_invalidate` over the `insight_edges` DAG + 5-min sentinel detection + reanalysis loop + confirmation-thresholded promotion (≥3 confirmations → semantic, ≥5 usages → procedural) + `max_staleness` search filter.
- **Reality (now):** schema is deployed (this session), but (a) **no configured sentinel uses the `invalidate` action**, so `invalidated_at` is never written; (b) episodic is empty (embedding down). The staleness layer won't actuate until the embedding fix lands + an invalidate path is wired.

### Remediation completed this session
- **PR #668 (open):** H5 (consolidator pagination) + M8 (sentinel evaluator allowlist) — TDD red-first, 102 memory tests green, ruff clean.
- **PR #672 (open):** deploy applies migrations via `docker exec supabase-db` (no `SUPABASE_DB_URL` needed) + covers `database/memory/`; forwards `OPENAI_API_KEY`. Scoped to `migrations/`+`memory/` (see reconciliation note).
- **Droplet migrations applied + tracked** (`public.schema_migrations`, bind-mount-persistent): `database/memory/` 014→029 (+ created missing `e2i_readonly`/`e2i_service` roles from `database/audit/011`); `database/migrations/` 034→057; `database/ml/` **016 (HPO), 020 (A/B testing), 028 (cohort constructor)**.

### ⚠️ Finding — the droplet migration state is INCONSISTENTLY PARTIAL
Attempting to scour **all** `database/` dirs revealed several subsystems were never deployed AND several migrations are **half-applied** (a prior run created an enum/type but errored before its table, with no transaction wrapper). Verified by direct existence checks:
- **Need per-migration surgery** (CORRECTED by per-object verification — earlier table-name guesses were wrong): `chat/036` user_roles is **FULLY APPLIED** (enum + `role` column + index + `role_level`/`has_role` functions all present), just untracked · `ml/013` tool_composer is **FULLY APPLIED** (all 6 tables, 4 enums, 2 triggers, 13 seed rows, 4 views, 5 functions present) but its committed file is latently broken for a fresh deploy · `ml/017` model_monitoring is **HALF-APPLIED** (`ml_drift_history` + 3 enums present; `ml_performance_metrics`/`ml_monitoring_alerts`/`ml_monitoring_runs`/`ml_retraining_history` + `alert_status_enum` MISSING) · `ml/021` ab_results is **NOT APPLIED** (collides with `ml/012`'s `calculate_fidelity_grade`) · `audit/012` security_audit_log is **NOT APPLIED** (its RLS policy references a `user_roles` table + `security_admin` role that exist **nowhere** — there is no `user_roles` table in the project; RBAC lives on `chatbot_user_profiles.role`).
- **Superseded (intentionally NOT applied — "use new chat schema"):** `chat/001` user_profiles, `chat/008` chat_messages → replaced by `chatbot_conversations`/`chatbot_messages` (029, present).
- **Process note (self-correction):** a regex table-name "baseline" shortcut produced **false baselines** (e.g. wrongly marked `chat/036` applied). Those records were deleted by timestamp; `schema_migrations` now reflects **only verified-applied state**. Lesson re-learned: data-driven per-object verification, never regex guesses.
- The all-dirs scour is therefore **deferred** — PR #672 is scoped to the two clean dirs so it is mergeable and stops the deploy-skip now, without aborting the deploy on a half-applied file.

### 📋 RECONCILIATION PLAN — deliberate, data-driven
**Methodology for every item (your standing instruction):** isolated **git worktree** + **TDD red-first** (a failing test/repro asserting the target state — e.g. the table exists and the migration re-runs idempotently) + **ralph-loop** to drive to green + **codex:codex-rescue → fixed point (ACCEPT)**. No bulk automation; each migration reconciled with verified evidence in the faithful environment.

1. **Reconcile the half-applied feature migrations** (one worktree each):
   - `audit/012` + `chat/036` (**compliance — prioritize**): repair the half-applied `user_role` enum, create `user_roles`, then apply `security_audit_log`.
   - `ml/013` tool_composer: guard/repair `routing_pattern`, create the missing tables idempotently.
   - `ml/017` model_monitoring: resolve the missing `m.name` dependency, then create `ml_monitoring_*`.
   - `ml/021` ab_results: reconcile the conflicting function (matching signature), then create `ab_experiment_results`/etc.
   Red-first test per item: assert the target objects exist AND the migration is re-runnable (idempotent).
2. **Rigorously re-baseline the remaining dirs** (core/ml/causal/chat/rag/audit) by **per-object** existence verification (every CREATE TABLE/TYPE/FUNCTION/INDEX/POLICY), not table-name regex — apply genuine gaps, baseline confirmed-applied, making each migration idempotent as it is touched.
3. **Re-enable the all-dirs scour** in `run_migrations.sh` (one-line change) once every dir is reconciled + cleanly baselined; verify `--dry-run` reports 0 pending.
4. **Activate embeddings:** merge #672 → deploy (forwards `OPENAI_API_KEY` + recreate) → verify `embed(...)` returns 1536-dim in `e2i_api` and an episodic write persists. Unblocks the lifecycle data flow.
5. **Wire an invalidate path** (an `invalidate`-action sentinel or an overturn hook) so `invalidated_at` is actually written and the staleness/reanalysis loop actuates.
6. **Open memory code fixes:** H1/H2 (route cross-tenant + Cypher write-injection) are the parallel API session's territory — now merged to `main`; **verify the gaps were actually closed**. H4 (blocking sync IO) needs a faithful load benchmark before fixing (#504 lesson).

Each step verified in the faithful environment (droplet `docker exec`), evidence recorded, before "done" is claimed.

### ✅ RECONCILIATION — Step 1 COMPLETE (branch `fix/reconcile-partial-migrations`, commit `57bc91dd`)

All 5 partial-state migrations were reconciled with the mandated methodology — isolated git worktree + TDD red-first + ralph-loop + codex:codex-rescue. A new faithful, isolated regression harness `scripts/test_migration_idempotency.sh` (throwaway DB in the **same** PG 15.8 instance, real `vector`/`pgcrypto`/`auth.uid()`; each migration applied **twice**) drove the red→green loop. The investigation **corrected** every earlier table-name guess and uncovered real latent defects:

| Migration | Droplet state (verified) | Reconciliation |
|---|---|---|
| `chat/036` | FULLY APPLIED, untracked | idempotent `CREATE TYPE` (DO/EXCEPTION) + `ADD COLUMN IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` |
| `audit/012` | NOT APPLIED; RLS policy referenced a non-existent `user_roles` table + `security_admin` role; illegal `timestamp` fn return-column | **RBAC repoint** to the real `chatbot_user_profiles.role` (reason-before-rules: the project never had a `user_roles` table); idempotent `DROP POLICY IF EXISTS`; `timestamp`→`event_timestamp` |
| `ml/013` | FULLY APPLIED, untracked; **file latently broken for fresh deploy** | idempotent types/tables/indexes/triggers + `ON CONFLICT` seeds; fixed `round(double precision,2)` (cast `::numeric`), an SRF-in-aggregate view, and an aggregate-in-recursive-CTE function — reconciled to the **deployed-working** definitions |
| `ml/017` | HALF-APPLIED (4 tables + `alert_status_enum` missing) | idempotent triggers; `ml_model_health_dashboard` used `m.name` but prod's column is `model_name` — fixed; re-apply fills the gap |
| `ml/021` | NOT APPLIED | idempotent triggers; renamed `calculate_fidelity_grade`→`ab_calculate_fidelity_grade` so it no longer clobbers `ml/012`'s canonical `fidelity_grade`-enum function used by `twin_fidelity_tracking` triggers |

**Two-layer verification, both GREEN:**
1. **Fresh harness** (fresh-deploy validity + idempotency): `ALL MIGRATIONS IDEMPOTENT & COMPLETE`.
2. **Faithful prod re-apply dry-run** — each reconciled file run inside `BEGIN … ROLLBACK` against the **real droplet** `postgres` DB (prod untouched): exit 0 for all 5. This caught two prod-only failures the fresh harness could not (the `m.name`/`model_name` drift and the `calculate_fidelity_grade` return-type collision).

**Key reusable lessons:** the committed migration files had **drifted** from the deployed-working definitions (views/functions hand-fixed on prod, never back-ported) — a fresh harness proves fresh validity but NOT re-apply onto prod's drifted objects, so the `BEGIN…ROLLBACK`-on-prod dry-run is the decisive faithful test. `round(double precision, integer)` does not exist in PG; a set-returning function cannot sit inside an aggregate; an aggregate cannot appear in a recursive CTE's recursive term; `timestamp` is legal as a column name but **not** as a function return-column name.

**Adversarial review (fixed point):** codex was account-blocked (`gpt-5.3-codex` not supported on a ChatGPT account), so two **independent** reviewers verified directly against prod — the resumed codex-rescue agent and a `code-reviewer` agent. Both returned **ACCEPT-WITH-NOTES, no blockers**, and converged on the same notes. Cheap high-value notes were addressed (documented the `security_admin`/inline-vs-`has_role` choice in audit/012; hardened the harness scaffold to the real `model_stage_enum`). Deferred-with-reasoning (pre-existing, out of reconciliation scope): `security_audit_log.user_id` has no UUID CHECK; 3 out-of-band `tool_registry` rows. One review claim was independently **corrected** — `v_classification_accuracy` DOES exist on prod (all 4 ml/013 views present).

**✅ APPLIED to the droplet + tracked (verified):** all 5 applied atomically (`--single-transaction`), recorded in `public.schema_migrations` (`audit/012…`, `chat/036…`, `ml/013…`, `ml/017…`, `ml/021…`). Post-apply verification: `security_audit_log` + 3 policies created; ml/017's 4 missing tables + `alert_status_enum` + `ml_model_health_dashboard` created (**half-applied gap filled**); ml/021's 3 `ab_*` tables + `ab_calculate_fidelity_grade` created, and ml/012's canonical `calculate_fidelity_grade`→`fidelity_grade` **left intact** (twin triggers unaffected).

**Deferred to a follow-up (NOT this PR):** ~~re-enabling the all-dirs scour (plan step 3)~~ — **[RESOLVED in PR #672 — see "Steps 2–4 COMPLETE" below.]** `run_migrations.sh` stayed scoped to `migrations/`+`memory/` at the time of #676 because the **rest** of the `audit/`/`chat/`/`ml/` dirs (e.g. `ml/012/014/015/018/019/022-027/029`, `audit/011`, superseded `chat/001/008`) were not yet verified-idempotent; adding a whole dir to the scour before reconciling every file in it would re-introduce the abort risk. Those dirs were subsequently reconciled in #682, so the scour was safely re-enabled in #672.

---

### ✅ RECONCILIATION — Steps 2–4 COMPLETE (PR #682 + PR #672, admin-merged 2026-06-04)

**Step 2 — re-baseline the remaining dirs (PR #682, merged `c93e732b`).** All 33 untracked migrations across core/ml/causal/chat/rag/audit reconciled re-apply-safe and applied/baselined to the droplet via **per-object** existence verification (not table-name regex): **28 APPLIED+tracked**, **4 BASELINE-only** (`core/030` superseded-taxonomy, `chat/002` dead-old-schema, `ml/015` CONCURRENTLY-already-applied, `ml/011` fully-applied-reconciled-to-prod). Deployed the `ml` causal-discovery schema (`ml/026` + `GRANT USAGE ON SCHEMA ml`). Two reusable tools committed: a dollar-quote-aware SQL idempotency transformer (`scripts/_idempotent_migration.py`) + a faithful `BEGIN…ROLLBACK`-on-prod re-apply harness (`scripts/test_migration_reapply.sh`). **Critical near-miss caught by adversarial review:** `ml/rollback_023.sql` is a DESTRUCTIVE utility (DROPs the **live** `estimator_evaluations` table written by `mlflow_tracker.py`) — `rollback_*`/`*_rollback`/`*_validation_queries` must be EXCLUDED from any runner/sweep. Also reusable: a **new schema needs `GRANT USAGE` before any table grant works** (`public` grants USAGE to anon/authenticated/service_role by default; a fresh schema does not).

**Step 3 — re-enable the all-dirs scour (PR #672, merged `19ee3239`).** `run_migrations.sh` `MIGRATION_DIRS` extended from {`migrations/`, `memory/`} to **all 7 dirs** (now that Steps 1–2 reconciled+tracked them), excluding `rollback_*`/`*_rollback`/`*_validation_queries`. **Decisive faithful test: a real-droplet `--dry-run` over all 7 dirs vs the live `schema_migrations` = "No pending migrations"** — the deploy scour is a verified clean no-op (every in-scope file already tracked; only genuinely-new files apply). An adversarial review workflow then found **2 confirmed MED apply-time defects** (both reproduced in ephemeral PG), fixed red-first in the same PR: (1) a future `CREATE INDEX CONCURRENTLY` would run under `--single-transaction` and **abort the deploy** → un-wrap heuristic broadened from only `ALTER TYPE ADD VALUE` to also bare `CONCURRENTLY` + self-`COMMIT` files (comment-stripped detection, erring toward un-wrap); (2) a self-`COMMIT` file could be **committed-but-untracked → wedge** the next deploy → the `schema_migrations` tracking row is now recorded in a **separate** invocation only after a clean exit (a post-COMMIT failure leaves the file UNTRACKED → idempotent retry). New TDD: `scripts/test_migration_scour_coverage.sh` (dir coverage + exclusion safety) + `scripts/test_migration_apply_txn.sh` (apply-time txn handling). Both deterministic on `postgres:15-alpine` — the `supabase/postgres` prod image **restarts mid-init** (its `pg_isready` healthcheck passes before the restart), racing a write-then-read test; the runner logic under test is vanilla-PG txn semantics, and the faithful-to-prod check is the separate real-droplet dry-run.

**Step 4 — activate embeddings (PR #672, merged) — ⏸️ PROD DEPLOY HELD (per user instruction).** Forwarded `OPENAI_API_KEY` to the api/worker containers via `x-common-env` (`OPENAI_API_KEY: ${OPENAI_API_KEY:-}`). Diagnosis verified at every config layer + a real embed call: `E2I_ENVIRONMENT=local_pilot` → OpenAI `ada-002` (1536-dim) is the **sole functional** embedding primary — **there is no Bedrock** (the `aws_production` branch is dead code; the `boto3` runtime isn't even installed, only mypy stubs). Absent the key, the 384-dim local sentence-transformers fallback mismatches the `episodic_memories.embedding vector(1536)` column on insert (or hard-fails "All embedding services unavailable") — either path starves every episodic write, and either way forwarding the key fixes it. The key resolves from the same `./.env` (deploy `cd $PROJECT_DIR` → compose auto-load) that already supplies the working `ANTHROPIC_API_KEY`; a **real OpenAI embed with the droplet key returned 1536-dim** (key valid + dim matches). The merge **stages** the fix on `main`; **the "Deploy to Production" workflow is `disabled_manually`, so prod still runs the old image until a deploy is triggered.** **OWED at deploy time (held):** verify `episodic_memories` populates with 1536-dim vectors and the consolidation/crystallization lifecycle flows.

**Remaining plan steps (unstarted):** Step 5 — wire an `invalidate`-action sentinel/overturn hook so `invalidated_at` is actually written and the staleness loop actuates (`config/sentinels.yaml` has no `action: invalidate` producer). Step 6 — verify the parallel API session's H1/H2 fixes (route cross-tenant PHI + Cypher write-injection), now merged to `main` (incl. the ws-auth PRs), actually closed the gaps.

---

## Snapshot

- Reviewed against `main @ 43345c84`, plus the **in-flight branch fixes** from the parallel API session: `fix/api-graph-methods-cypher @ 69f0961d` and `fix/api-idor-cognitive-memory @ 9e08690a`. The post-fix versions of `src/memory/semantic_memory.py`, `src/api/routes/graph.py`, `src/api/routes/memory.py`, `src/api/routes/cognitive.py` were read via `git show <sha>:<path>` (never checked out — the review could not disturb the active API session).
- Scope: the 28 core files of `src/memory/` (~15.3K LOC), the `database/memory/` schemas, and the memory/cognitive/graph route layer.
- Findings: 64 raw → **61 confirmed** by independent adversarial verifiers, **3 refuted**, deduped to **35 distinct issues** (6 HIGH, 9 MEDIUM, 20 LOW) regrouped by adversarially-adjusted severity.

## Methodology

11 module reviewers + 3 cross-cutting specialists (security/PHI, doc-drift, test-integrity) produced findings; each HIGH/MEDIUM finding was then handed to an independent **adversarial verifier** (default-to-false-positive) that re-read the cited code at the pinned SHA. A synthesis pass deduped and regraded; a completeness critic flagged what was not covered.

## Dispatcher independent verification (added on top of the workflow)

Per project discipline ("verify before reporting plan results"), every HIGH was re-confirmed against the actual code by the dispatcher. Results:

- **H1 (cross-tenant reads), H5 (pagination):** confirmed verbatim — `search_memory` lacks the brand-gating block that `list_episodic_memories` has in the same file; `grep` for `.limit(`/`.range(`/`.order(` in the consolidator returns nothing.
- **H2 (Cypher write injection):** confirmed, with the precise live vector pinned — `rag/cognitive_backends.py:297` passes LLM-derived `relationship_type.upper()` (`.upper()` does not neutralize `]`/`(`). The ~20 agent-hook callers pass **trusted `E2IEntityType` enums**, so they are *not* the injection vector. The `004_cognitive_workflow.py` path cited by one reviewer is **dead code** (see refutations), so H2's reachability rests on the `cognitive_backends` LLM path. This is a second-order (LLM-mediated) injection, cheap to fix with the existing read-side allowlist machinery.
- **H3 (sync RedisSaver):** code path confirmed — `working_memory.get_langgraph_checkpointer()` returns a sync `RedisSaver` and is called by the live `chatbot_graph.py:1927` and `explainer/graph.py:40`. **Caveat:** the library-level `NotImplementedError` depends on prod-pinned deps (`langgraph 1.2.0` + `langgraph-checkpoint-redis 0.4.1`); the **local `.venv` is stale** (langgraph 1.0.5, no redis checkpoint) and would *falsely pass* a naive local repro via the `ImportError → MemorySaver` branch. A faithful regression test requires the pinned deps + a Redis in CI.
- **H4 (blocking sync IO):** confirmed reachable on the FastAPI request loop; the **magnitude** (p99 under load) is an unbenchmarked hypothesis. The consolidator variant of this class was correctly **refuted** (it runs under `asyncio.run` in Celery, owning its own loop).
- **M8 (sentinel evaluator injection):** confirmed — `_validate_pattern_config` allowlists `table` only for `invalidation_count`; `threshold_breach`/`freshness` validate key-presence only, then interpolate `table`/`column` raw. (Reviewer rated HIGH; lead downgraded to MEDIUM because it requires an authenticated, brand-scoped operator role.)

---

## HIGH severity

### H1 — Cross-tenant PHI leakage on the memory READ surfaces (search / cognitive / semantic-paths)
- **Subsystem:** API routes (memory, cognitive) — *API-session territory*
- **File:** `src/api/routes/memory.py:305-373` (search), `:612-660` (semantic paths); `src/api/routes/cognitive.py:366-371` (query) — post-fix `@ 9e08690a` / retriever `@ 69f0961d`
- **Problem:** The IDOR fix added `require_viewer` + `_brand_allowed`/`get_user_brands` scoping to the episodic list/get/create endpoints (`memory.py:471-484, 540-542, 398-399`) but did NOT apply it to three higher-value surfaces. `search_memory` passes caller-controlled `request.filters` straight into `hybrid_search` with no grant check; the RPC predicate `filters->>'brand' IS NULL OR em.brand = filters->>'brand'` means omitting `brand` returns ALL brands and `filters.brand=BrandB` returns BrandB. It then returns raw `r.content` and FULL `r.metadata` (not the redacted `EpisodicMemoryResponse`). `query_semantic_paths` accepts arbitrary `start_entity_id`/`kpi_name` and traverses the FalkorDB Patient/HCP causal/influence graph with zero tenant scope (BOLA). `cognitive.query` forwards `request.brand` unchecked.
- **Evidence:** `memory.py:316-342` `hybrid_search(..., filters=request.filters)` with no `get_user_brands` (contrast `:472-484`); `database/memory/011_hybrid_search_functions_fixed.sql:120` null-brand-means-all; `src/rag/retriever.py:279-281` graph results not brand-filtered; `require_viewer` is authn-only (`src/api/dependencies/auth.py:333`). Reported independently by two reviewers and dispatcher-confirmed.
- **Recommendation:** Mirror `list_episodic_memories`: for non-admins inject the caller's grant into `filters['brand']` server-side (reject/empty out-of-grant brands) BEFORE calling `hybrid_search`, on BOTH `search_memory` and `cognitive.query`; resolve the owning brand of `start_entity_id`/`kpi_name` (or post-filter returned nodes to the grant set) for `query_semantic_paths`; scope the entities/kpi graph-traversal path. Nodes must carry a `brand` property to filter on. Return 404 for out-of-grant ids to avoid existence leakage.

### H2 — Write-side Cypher injection into the PHI graph (`add_e2i_relationship` / `add_e2i_entity`)
- **Subsystem:** Semantic memory (FalkorDB) — *API-session territory (graph branch)*
- **File:** `src/memory/semantic_memory.py:376-396` (rel), `258-270` (entity) — branch `@ 69f0961d`
- **Problem:** Read paths were allowlisted (`_validate_relationship_types`/`_validate_node_labels`, used at `:955/999/1064/1119/1203/1370`), but the WRITE methods splice caller-supplied `rel_type` into `MERGE (s)-[r:{rel_type}]->(t)` (`:380, :388`) and interpolate `label` + property-KEY names (`:262-270`) with no validation. **Live vector (dispatcher-confirmed):** `rag/cognitive_backends.py:297` passes LLM-derived `relationship_type.upper()` (`.upper()` does not block injection); `graph.query(q, params)` binds only values, not structural tokens. Agent-hook callers pass trusted `E2IEntityType` enums and are not the vector. Same DB and threat model PR #657 hardened on reads; a duplicate of the bug lives in the dead `006_memory_backends_v1_3.py:1278`.
- **Recommendation:** Call `_validate_relationship_types(rel_type)` / `_validate_node_labels(label)` before splicing, validate property keys (allowlist or strict identifier regex), and add a negative test asserting a poison rel_type/key raises `ValueError` and never reaches the recorded query.

### H3 — Async LangGraph graphs compiled with the SYNC RedisSaver → cross-request memory silently broken when Redis is up
- **Subsystem:** Working memory / checkpointer — *independent (being drafted)*
- **File:** `src/memory/working_memory.py:87-126` (and parallel `src/memory/langgraph_saver.py:56` `create_checkpointer()`)
- **Problem:** `get_langgraph_checkpointer()` constructs the synchronous `RedisSaver` (`working_memory.py:110-111`). Consumers run async: `chatbot_graph.py:1929` compiles `e2i_chatbot_graph` with it and invokes `await ...ainvoke(...)` (`:1998`) / `astream(...)` (`:2162`); `explainer/graph.py` similarly. langgraph 1.2.0 `AsyncPregelLoop` awaits `checkpointer.aput_writes`/`aput`/`aget_tuple` (`_loop.py:1739/1767/1862`). The 0.4.1 sync `RedisSaver` implements only sync `get_tuple/put/put_writes/list`; the async methods resolve to `BaseCheckpointSaver`'s `raise NotImplementedError`. So every checkpoint persist/read during an async turn raises — exactly when Redis is reachable. When Redis is DOWN, the `except` falls back to `MemorySaver` (which has async methods), masking the bug in dev/CI.
- **Evidence:** Confirmed against PRODUCTION-pinned deps (requirements.lock: langgraph==1.2.0, langgraph-checkpoint-redis==0.4.1). The correct `create_async_checkpointer` → `AsyncRedisSaver` exists but has ZERO `src/` callers. NOTE: the local `.venv` is stale (langgraph 1.0.5, no redis checkpoint) and would FALSELY pass a naive local repro by hitting the ImportError→MemorySaver branch.
- **Recommendation:** For async graphs return an `AsyncRedisSaver` and `await asetup()` (make the factory async + await it from chatbot/explainer, or build/cache the AsyncRedisSaver in the FastAPI lifespan and inject it). Fix BOTH `working_memory.get_langgraph_checkpointer()` and `langgraph_saver.create_checkpointer()`. Add a regression test that compiles a tiny StateGraph with the REAL RedisSaver against a live/fake Redis on the PINNED deps and asserts `ainvoke` round-trips a checkpoint.

### H4 — Blocking synchronous Supabase/LLM/embedding/FalkorDB IO inside `async def` coroutines stalls the event loop
- **Subsystem:** Episodic / procedural / semantic memory + services factories (systemic) — *independent (episodic/procedural/factories portion being drafted)*
- **File:** `src/memory/services/factories.py:88,103,150,202-227,403,414,462`; `src/memory/episodic_memory.py:294,467` and broad set `271-863`; `src/memory/procedural_memory.py:125,237,371`; `src/memory/semantic_memory.py` `self.graph.query(...)` from async
- **Problem:** Every `async def` memory function uses the SYNCHRONOUS Supabase client (`get_supabase_client()` → `create_client`, `factories.py:570-610`) and the sync LLM/embedding SDKs and calls `.execute()`/`.create()` with no `await`/`to_thread`/`run_in_executor`. Each call blocks the single FastAPI worker for the full network round-trip, serializing concurrent requests. The codebase already wraps this pattern elsewhere (`feast_client.py:1098 await asyncio.to_thread(...)`, `dispatcher.py:357 loop.run_in_executor(...)`) — the memory module simply doesn't. This is the documented #504 incident class.
- **Evidence:** Reachable on the live loop: `memory.py:344/372` route handlers `await insert_episodic_memory_with_text`; ~10 agent memory_hooks await searches/inserts; `chatbot_graph.py:1558`. Async clients exist purpose-built (`get_async_supabase_client`, `factories.py:638`) and are unused here.
- **Recommendation:** Use the async SDK clients (`openai.AsyncOpenAI`, `anthropic.AsyncAnthropic`, `get_async_supabase_client`) and `await` them, OR wrap blocking calls in `await asyncio.to_thread(...)` (including `SentenceTransformer.encode`). NOTE: the p99 magnitude is an unverified hypothesis until benchmarked.

### H5 — Un-paginated episodic SELECTs in the consolidator silently truncated by PostgREST's 1000-row cap
- **Subsystem:** Lifecycle / consolidator — *independent (being drafted)*
- **File:** `src/memory/lifecycle/consolidator.py:650-662` (deduplicate_episodic), `1374-1383` (extract_procedural_templates)
- **Problem:** Both SELECT episodic candidate rows with only `.eq`/`.is_` filters — no `.limit()`/`.range()`/`.order()` (grep-confirmed no pagination anywhere in the module, and no `db-max-rows`/`PGRST_DB_MAX_ROWS` override in the repo). The `extract_procedural_templates` path has NO `IS NULL` filter, so its candidate set is every still-present row for the brand and grows monotonically with the (never-pruned) table; once it crosses 1000 the server silently returns a subset, and clusters straddling the boundary undercount `SUM(dedup_counter)` → wrong `effective_cluster_size` (`:1438`) → wrong template-extraction gate and downstream semantic-promotion threshold. Runs daily with `brand=None` (whole portfolio) via the Celery beat schedule (`celery_app.py:348-352` → `insight_lifecycle_tasks.py:50` → `consolidator.py:1849` → `run(brand=None)`); per-brand handling at `:518-557` is metrics-only and does not re-scope the SELECTs.
- **Evidence:** Verified verbatim: `:650-662` only `.eq`/`.is_` then `query.execute().data`; `:1374-1383` only optional `.eq("brand")`. The dedup path is partly self-bounding (IS NULL backlog usually < cap); the template path is the unbounded one driving HIGH.
- **Recommendation:** Paginate via `.range()` until exhaustion (or keyset on `occurred_at`), asserting a short final page before treating a group as complete; or process per-(brand,signature) with server-side bounds.

---

## MEDIUM severity

- **M1 — Embedding dimension mismatch (384 vs 1536) with no guard.** `factories.py:230-312`; consumers `episodic_memory.py:327-336,494-500` / `procedural_memory.py:260-263`. Default `E2I_EMBEDDING_FALLBACK='true'` → on a primary-API outage `LocalEmbeddingService` emits 384-dim into `vector(1536)` columns; pgvector rejects the write (loud error). In agent hooks the error is swallowed (memory write silently LOST during the outage); on routes it becomes HTTP 500. The resilience fallback fails to serve its purpose precisely when it activates. **Fix:** validate `len(embedding) == vector_dims` before write and fail-fast/skip-persist, or refuse writes while `is_using_fallback`.
- **M2 — Sync Supabase client blocks the loop in the FastAPI-reachable `/crystallize` route.** `crystallizer.py:149-172,296-377,415-430,485`, reachable via `executive_insights.py:302-316`. MEDIUM (operator-only, low-frequency). **Fix:** async client / `to_thread` / offload to Celery.
- **M3 — `_promote_to_semantic` N+1 (one episodic SELECT per candidate causal_path).** `consolidator.py:1236-1278`. Runs only in the offline daily sweep (dedicated loop), so MEDIUM-leaning-LOW. **Fix:** batch via `.in_("causal_path_id", path_ids)`.
- **M4 — `count_memories_by_type` ignores `days_back`; `/memory/stats` reports all-time count as "recent 24h".** `episodic_memory.py:837-863`, consumer `memory.py:607-608,631-635`. User-visible wrong metric. **Fix:** add `query.gte("occurred_at", cutoff)`.
- **M5 — `config.py` LLMConfig/EmbeddingConfig parsed from YAML but never wired into the factory (silent config-drift).** `config.py:79-98,228-248` vs `factories.py:814-857`. Operator edits to YAML model/dimensions have NO effect. **Fix (reason-before-rules):** wire the factory to honor `MemoryConfig`, or delete the unused fields after confirming intent.
- **M6 — `get_embedding_service()` returns a fresh uncached instance every call**, defeating the per-instance cache AND the `FallbackEmbeddingService` 5-minute primary-retry backoff (reset every call → under an outage every op re-attempts the down primary). `factories.py:814-869`. **Fix:** cached singleton keyed on `(environment, use_fallback)`, or module-level state.
- **M7 — Cross-brand IDOR on PATCH/DELETE sentinel routes.** `src/api/routes/sentinels.py:234-258` — *API-session territory.* GET/create enforce brand membership; PATCH/DELETE (`require_operator` role-only) were missed → a Brand-X operator can disable/delete a Brand-Y sentinel by id enumeration. **Fix:** apply `get_sentinel`'s brand check before `.update()/.delete()`; 404 out-of-grant.
- **M8 — `threshold_breach`/`freshness` sentinel evaluators interpolate unvalidated `table`/`column` into PostgREST.** `src/memory/sentinels/registry.py:324-357` — *independent (being drafted).* Operator-supplied `pattern_config`; `_validate_pattern_config` checks only key-presence + `op`. PostgREST projection mini-language interprets `*`, resource embedding `fk_table(ssn,dob)`, aliasing — confirmed the installed client only strips whitespace. A brandless `table` defeats `.eq("brand", brand)` scoping. **Fix:** per-table permitted-column allowlist validated at registration + evaluation; require a `brand` column; reject PostgREST metacharacters.
- **M9 — PHI-bearing memory tables granted to `authenticated` with no RLS.** `database/memory/001_agentic_memory_schema_v1.3.sql:763-769` (`episodic_memories`, `semantic_memory_cache`, `learning_signals`). RLS IS used on peer tables (chat/002, ml/014, audit/012) — an inconsistency, not a convention. MEDIUM: app reaches these via the backend client, so exploitation requires an end-user JWT hitting PostgREST directly (depends on the self-hosted Docker topology — confirm with owner). **Fix (reason-before-rules — separate from the documented API auth-fail-open decision):** decide whether end-user JWTs reach these tables; if yes add RLS + brand policies mirroring `ml/014`, else REVOKE the `authenticated` grant.
- **M10 — Sentinel enters cooldown even when every action failed → silently suppresses matching alerts.** `registry.py:616-732`: `last_fired_at` bumped unconditionally after the match loop, never consulting `result.actions_taken`. **Fix:** anchor `last_fired_at` on real success only.
- **M11 — `006_memory_backends_v1_3.py` is a vestigial divergent duplicate (drift hazard).** ~2074 LOC, non-importable (leading-digit filename + nonexistent config path), ZERO `src/` consumers, but a PRE-FIX snapshot still carrying broken parameterized var-length Cypher (`:1294`), missing LIMITs, a `bulk_insert` that drops 6 of 11 FK refs, and the read/write injection surface. Its tests use a `MagicMock` graph (false-green). **Fix (reason-before-rules):** classification is vestigial-DELETE — confirm intent via PR/issue history before removing module + its two test files.
- **M12 — `004_cognitive_workflow.py` is orphan code importing non-existent modules; tests false-green via `sys.modules` injection.** Neither `src/memory/memory_backends.py` nor `agent_registry.py` exists → ImportError on first invocation; production uses `create_dspy_cognitive_workflow` in `src/rag/`. **Fix (reason-before-rules):** DELETE 004 + its test if confirmed dead, else repoint imports.
- **M13 — `_promote_to_procedural` ignores `brands_with_dedup_errors`, diverging from the documented short-circuit contract.** `consolidator.py:1636-1664`. No current harm (procedural promotion keys off usage/success, not the inconsistent counter). **Fix:** narrow the generic docstrings to match the producer, or add the skip.

---

## LOW severity (compact)

| ID | File:line | Problem | Recommendation |
|---|---|---|---|
| L1 | `episodic_memory.py:176-189,274-294` | `EpisodicSearchFilters.min_importance`/`days_back` declared but never forwarded to the search RPC (silent filter-drop; no current caller sets them) | Remove unsupported fields or add RPC params |
| L2 | `procedural_memory.py:199-211,339-373` | Non-atomic read-modify-write of usage/success counters → lost updates under concurrency | Server-side atomic increment RPC |
| L3 | `signals.py:100-109,179-184` | `iter_messages` `block_ms=0` could busy-spin a future consumer (scaffolded, no consumers) | Require `block_ms>0` or add backoff |
| L4 | `evidence_cache.py:117-127` | Eviction `_max_size // 10 == 0` for `max_size<10` → no eviction (prod default 1000) | Floor eviction to `max(1, ...)` |
| L5 | `cognitive_integration.py:322-342` | `process_query` stores context under a different uuid than messages (HIGH→LOW: no prod consumer; `cognitive.py` is the live impl) | Pass `session_id` into `create_session` |
| L6 | `cognitive_integration.py:773-779` | `_store_to_graphiti` uses naive timestamp | `datetime.now(timezone.utc).isoformat()` |
| L7 | `crystallizer.py:159-172` | Candidate SELECT no LIMIT/pagination → non-deterministic provenance sha256 at scale | Paginate via `.range()` |
| L8 | `invalidator.py:161-174` | `record_hit()` unconditional regardless of UPDATE match → over-counted `invalidated_by_type` (metrics-only) | `if upd.data: record_hit(...)` |
| L9 | `consolidator.py:517-557` | `brand=None` fanout attributes all templates to `touched_brands[0]` + global denominator per brand → wrong per-brand metrics | Track per-brand in `result.by_brand` |
| L10 | `consolidator.py:1660-1664` | Empty `applicable_brands` short-circuits the brand-scope skip (unreachable today — write paths coerce to `['all']`) | Tighten the skip condition |
| L11 | `memory.py:467-487` | Non-admin omitting `?brand` pinned to `allowed_brands[0]` → multi-brand users see only one brand | Loop/merge over grants |
| L12 | `cognitive.py:41-69` | `_caller_id` can fall through to `'anonymous'`, conflating principals (fail-closed today) | Treat missing id as 401/403 |
| L13 | `memory.py:570-622` | `record_procedural_feedback` (require_viewer) mutates any procedure's counts + injects learning signals, no ownership/brand check; `outcome` not enum-validated (reviewer: MEDIUM) | Fetch procedure, check grants; validate `outcome` |
| L14 | `factories.py:544-562,...` | Check-then-set singleton init race under concurrent async startup; orphaned client leaks httpx pool | Lock + `aclose()` on reset |
| L15 | `factories.py:384,478` | Stale model pins `claude-3-5-sonnet-20241022` (HIGH→LOW: identical cost, plausibly intentional) | Align or drive from shared config |
| L16 | `factories.py:535,555` | `get_redis_client` docstring says `:6379` but default `:6382` (6379 is FalkorDB's family) | Fix docstring |
| L17 | `factories.py:76,97-99,...` | Embedding cache keyed on `hash(text)` (per-process randomized, collision risk) + unbounded | Key on `hashlib.sha256`, bound |
| L18 | `e2i_extractor.py:197-234` | `extract_relationships` uses single `str.find` → only first occurrence (HIGH→LOW: zero prod consumers, scaffolded) | `re.finditer` with boundaries; fix before wiring |
| L19 | `auth_middleware.py:77-83` | Public graph endpoints echo full Patient/HCP `node.properties` (auth fail-open is the documented INTENTIONAL owner decision; residual is the PHI property echo) | Label/property allowlist excluding Patient/HCP |
| L20 | `011_hybrid_search_functions_fixed.sql:70-78` | HNSW + leftover IVFFlat indexes coexist on same columns → double write amplification | `DROP INDEX` the IVFFlat once HNSW chosen |
| L21 | `021_insight_lifecycle.sql:303-327` | `verify_insight_chain` ancestor-walk lacks `ml_prediction`/`executive_insight` branches → invalidated ancestor returns valid (latent) | Add `ELSIF` branches |
| L22 | `001b_add_foreign_keys_v3.sql:281-322` | `sync_hcp_patient_relationships_to_cache` counts only the first INSERT (undercounts; logging only) | Accumulate second-block ROW_COUNT |

---

## Doc-drift (answers "is ARCHITECTURE.md outdated?")

**More accurate than feared, with specific drift.** §5 Memory Subsystems correctly describes the consolidator/crystallizer/sentinels, but:

| Severity | File:line | Drift | Fix |
|---|---|---|---|
| LOW (was HIGH) | `README.md:419-421` | Working-memory TTL stated `3600s` (1h); actual config is `86400s`/24h (`config.py:181`, `005_memory_config.yaml:49/60`); `ARCHITECTURE.md` correctly says 24h. README wrong vs code AND vs its own arch doc | "TTL: 86400 seconds (24 hours)" |
| MEDIUM | `docs/ARCHITECTURE.md:712-725` | §5.1 says `Consolidator.run()` runs "three steps" but omits the FOURTH phase `extract_procedural_templates` (`consolidator.py:497`, the #389 fix) — functionally load-bearing | Update to four steps |
| LOW | `docs/ARCHITECTURE.md:678-683,712-722,749-762` | Every `file:line` citation in §5 / its mermaid is stale (file grew during #388/#389): `run()` at `:448` not `:176`; `class Crystallizer` at `:104` not `:102`; etc. | Re-point references; prefer symbol names |
| LOW | `docs/ARCHITECTURE.md:458-471` | §4.3 headers "15 Edge Types" but the table lists only 10 (`graphiti_config.py` defines 11; ontology doc says 13+2) — three sources disagree | Complete the table or relabel |
| LOW | `database/memory/021_insight_lifecycle.sql:236-237` | Comment claims `verify_insight_chain` logs to `audit_chain_verification_log`, but the (STABLE) body has no INSERT; real logging is in Python `insight_verifier.py:261-286` | Fix comment |

Also: README's "Semantic Memory (FalkorDB)" omits the **Graphiti** layer (`graphiti_config.py` has `falkordb_host/port/password` — Graphiti runs *on top of* FalkorDB).

---

## Test-coverage (false-greens that mask findings)

A recurring pattern: tests mock the very dependency whose contract is broken, so the suite stays green while the production path fails.

1. **Checkpointer round-trip never tested (masks H3).** `test_langgraph_saver.py:29-40` stubs the saver with `MagicMock`/`AsyncMock` and asserts only which class was constructed. No test compiles a StateGraph and runs `await graph.ainvoke(...)` against the REAL RedisSaver. **Add** a fakeredis/real-Redis test on the PINNED deps in a Redis-available CI lane.
2. **006 FalkorDB tests are false-green (supports M11).** `test_memory_backends_v1_3.py:801-876` use a `MagicMock(result_set=[])` graph that absorbs any query and never parses Cypher. **Preferred:** delete the dead module + tests.
3. **004 cognitive-workflow tests pass via fake-module injection (masks M12).** `test_cognitive_workflow_v2.py:84-89` registers `sys.modules['src.memory.memory_backends']` — a module that does NOT exist. **Treat the current mock tests as non-evidence.**
4. **Extractor relationship tests are self-defeating (L18).** `test_e2i_extractor.py:338-409` wrap assertions in `if rels:` loops that no-op on empty lists.
5. **Brand-isolation tests use a FakeDB (relevant to M9/L19).** `test_brand_isolation.py:31-148` validates app-layer logic only; not evidence of DB-level tenant isolation.

---

## Refuted findings (3 — for transparency)

The adversarial verifiers correctly rejected these on reachability grounds (each was a real code smell but not a reachable production defect as stated):

1. **`langgraph_saver.py:28-82` duplicate sync-checkpointer** — REFUTED: `create_checkpointer`/`create_async_checkpointer` have ZERO production callers; the real production checkpointer is `working_memory.get_langgraph_checkpointer` (which IS the H3 bug). The `langgraph_saver.py` twin is test-only/dead.
2. **`004_cognitive_workflow.py:659-676` Cypher injection via the reflector path** — REFUTED: `004` is dead/unimportable (imports a nonexistent `memory_backends` module); the only caller of its `sync_to_semantic_graph` is the dead reflector. Production reflection uses `CognitiveService._run_reflector` (Graphiti `add_episode`). (The *live* injection vector is H2 via `cognitive_backends.py`, not this.)
3. **`consolidator.py` sync IO blocking the loop** — REFUTED: the sentinel/beat path runs under `asyncio.run` in a Celery prefork worker, owning its own event loop with no other coroutines to stall. (The episodic/routes variant — H4 — IS reachable on the shared FastAPI loop.)

---

## What this review did NOT cover (round-2 scope)

Strong depth on the `src/memory/` *core*; the **memory-consuming surface** was out of scope and is the natural next round:

1. **21 agent `memory_hooks.py`** (`src/agents/*/`, `src/agents/ml_foundation/*/`) — the primary integration seam; lazy-init fail-open-to-`None`, what each persists (incl. patient context), and whether free-text/PHI or attacker-influenced labels flow into episodic embeds / semantic MERGE.
2. **Celery task layer + beat scheduling** — `crystallization_tasks.py`, `insight_lifecycle_tasks.py`, `sentinel_actions.py`, `workers/celery_app.py`. The real drivers of consolidation/crystallization/invalidation; the documented non-atomic consolidator window is only assessable with the schedulers in view.
3. **Worker pub/sub consumers** — `workers/event_consumer.py`, `monitoring.py`. Is the invalidation signal actually consumed, or publish-only (fire-and-forget)?
4. **Write-side PHI** — what agent hooks embed/store; verify `patient_id`/`journey_id` are pseudonymous tokens and no raw PHI is embedded as searchable text or queryable graph properties.

---

## Coordination with the parallel API session

These findings live in files the API session is actively rewriting — they should be folded into its in-flight branches, NOT fixed here:

| Finding | File | API-session branch |
|---|---|---|
| H1 (cross-tenant search/cognitive/paths) | `routes/memory.py`, `routes/cognitive.py` | `fix/api-idor-cognitive-memory` |
| H2 (Cypher write injection) | `semantic_memory.py`, `cognitive_backends.py` | `fix/api-graph-methods-cypher` |
| M7 (sentinel route IDOR) | `routes/sentinels.py` | `fix/api-idor-explain-sentinels` |
| L13 (procedural-feedback BOLA) | `routes/memory.py` | `fix/api-idor-cognitive-memory` |

## Remediation status (this session)

Drafting fixes on `fix/memory-independent-highs` (isolated worktree) for the **independent HIGHs** that don't collide with the API session: **H5** (consolidator pagination), **H4** (blocking IO — episodic/procedural/factories), **M8** (sentinel evaluator allowlist), and **H3** (sync RedisSaver — implemented but faithful verification deferred to a Redis-available CI lane on pinned deps).
