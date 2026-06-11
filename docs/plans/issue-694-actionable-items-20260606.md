# Issue #694 Actionable Items — Implementation Plan

> **For agentic workers:** execution method (user-specified) = **per-shard git worktree isolation → TDD red-first → ralph-loop iteration → `codex:codex-rescue` review to ACCEPT (fixed point) → PR**. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the *faithfully-actionable* subset of the #694 memory-subsystem carry-forward ledger with real, tested fixes (no mocking of the unit under test).

**Architecture:** Three independent shards, each a standalone PR off latest `origin/main` in its own worktree (disjoint from open PR #763 = twin-H11, which only touches `digital_twin.py`). A ralph-loop drives each shard to a fixed point (red-first test → implement → codex-rescue ACCEPT → `<promise>`), gated by real targeted tests + ruff + the RPC↔DDL guard.

**Tech Stack:** Python 3.12, FastAPI, Supabase (PostgREST), FalkorDB (Cypher), pytest (`-n0` single-process on the memory-constrained droplet), ruff, mypy (CI is arbiter), Codex CLI (`codex:codex-rescue`).

---

## Scope & triage (REASON-BEFORE-RULES applied)

The #694 "Deferred" list is heterogeneous. After intent-investigation + faithfulness checks, the buckets are:

### ✅ IN SCOPE (faithfully actionable now)
| Shard | Item | Complexity | PR |
|---|---|---|---|
| **A** | **M3** — `_promote_to_semantic` N+1 (one episodic SELECT per candidate path) | S | standalone |
| **B** | **L-code-batch** — L1, L3, L5, L6, L7, L14, L18 (clean Python; +L2/L10 conditional on red-first) | M | standalone |
| **C** | **H1** — add `brand` to causal-graph nodes + brand-scope `query_semantic_paths`; relax fail-closed gate | L (security-sensitive) | standalone |

### 🅓 DEFERRED within scope (separate, careful)
- **D — L-migration-batch (L20, L22):** L20 = drop redundant IVFFlat (HNSW already covers it) — needs a migration **and faithful prod EXPLAIN-ANALYZE perf parity check** before the drop; L22 = `sync_hcp_..._to_cache` undercounts its return (logging-only) — a SQL `CREATE OR REPLACE` migration. Both touch `database/memory/` (applied out-of-band). Recommend a 4th PR *after* A–C, or defer L20 until a prod perf window.

### ❌ EXCLUDED — with reasons (do NOT build)
- **L21** — **ALREADY DONE** by `database/memory/030_verify_insight_chain_ancestor_types.sql` (commit `d39fb7d5`): adds the missing `ml_prediction` + `executive_insight` ancestor-walk branches; header notes "Reproduced faithfully on the droplet." The ledger's "needs a DB migration" is stale. **Action:** verify 030 is applied to prod (read-only, needs auth) + tick L21 in the #694 ledger. No code.
- **L9** (consolidator brand=None metrics denominator) — **owner decision, "do NOT re-flag"** (#694).
- **L11** (non-admin no-`?brand` pinned to first grant) — **owner decision, "do NOT re-flag"** (#694).
- **L12** (cognitive `_caller_id` → "anonymous" fallthrough) — covered by the **"auth fail-open" owner decision** (#694). Treat as excluded unless the owner reopens it.
- **L15** (model pins `claude-3-5-sonnet-20241022` not aligned with `MemoryConfig`) — likely *intentional* version-pinning; needs an owner intent call, not an autonomous fix. Document-only at most.
- **Blocked (track, don't block):** Step 5 (needs `executive_insights`>0; currently 0), H4 (needs faithful concurrent-load benchmark — #504 lesson), TestCov (needs a RediSearch Redis CI lane), M9 (owner topology decision). Cannot be done with real results now.

---

## File Structure (what each shard touches)

- **Shard A:** `src/memory/lifecycle/consolidator.py` (`_promote_to_semantic`, ~:1280-1328) · `tests/unit/test_memory/test_consolidator.py` (extend `FakeQuery.in_()` + execute-count tracking).
- **Shard B:** `src/memory/episodic_memory.py` (L1) · `src/memory/coordination/signals.py` (L3) · `src/memory/cognitive_integration.py` (L5 session_id, L6 tz-aware ts) · `src/memory/crystallization/crystallizer.py` (L7 pagination) · `src/memory/services/factories.py` (L14 singleton lock) · `src/memory/extractors/e2i_extractor.py` (L18 finditer) · matching tests under `tests/unit/test_memory/`.
- **Shard C:** `src/memory/semantic_memory.py` (node creation + `traverse_causal_chain:701` + `find_causal_paths_for_kpi:876`) · `src/api/routes/memory.py` (`query_semantic_paths:706` gate) · `tests/` (FalkorDB brand-scoping; needs a branded test graph fixture).

---

## Shard A — M3: batch `_promote_to_semantic` episodic lookups

**Why:** `consolidator.py:1307-1311` issues one `episodic_memories` SELECT per candidate `causal_path` inside the loop (`:1280`). N candidates ⇒ N+1 queries. Offline daily sweep, so MEDIUM-leaning-LOW, but a clean, zero-behavior-change batching win. The `.in_()` batch pattern already exists at `consolidator.py:1007`.

**Faithfulness:** query-count assertion via a real `FakeSupabase` test-double that records `execute()` per table (the *consolidator* logic is the unit under test — not mocked), plus a behavior-equivalence test (identical promotion decisions old vs new).

**Files:** Modify `src/memory/lifecycle/consolidator.py`; Test `tests/unit/test_memory/test_consolidator.py`.

- [ ] **A1 — Red: N+1 → 1 query count test.** Add `test_promote_to_semantic_batches_episodic_queries`: build ≥5 candidate paths with valid confirmation counts; run `_promote_to_semantic(result, brand=None)`; assert the test-double recorded **exactly 1** `episodic_memories` SELECT (fails today at N). Extend `FakeQuery` with `.in_(col, values)` and add per-table `execute()` counting to the harness.
- [ ] **A2 — Run red.** `…/.venv/bin/python -m pytest tests/unit/test_memory/test_consolidator.py -k batches_episodic -n0 -p no:warnings` → expect FAIL (count==N).
- [ ] **A3 — Red: behavior-equivalence test.** `test_promote_to_semantic_batch_equivalence`: same fixture; assert the set of promoted `path_id`s + per-path confirmation counts are identical to the pre-batch expectation; include a dedup-error-skipped brand to lock `consolidator.py:1290-1295` semantics.
- [ ] **A4 — Implement batch.** Collect promotable `path_id`s (after the dedup-error brand filter), issue one `.in_("causal_path_id", path_ids)` SELECT, build `{path_id: rows}`, replace the in-loop per-path query with dict lookups. Preserve the optional `.eq("brand", brand)` scoping (`:1275-1276`). Guard the empty-`path_ids` case (skip the query). If `len(path_ids)` can exceed a safe bound, chunk by 500.
- [ ] **A5 — Run green.** same pytest selectors → PASS (count==1, equivalence holds).
- [ ] **A6 — codex-rescue gate.** `codex:codex-rescue` on the diff; brief MUST include the anti-mocking pushback paragraph (below). Resolve to ACCEPT.
- [ ] **A7 — Full targeted suite + ruff.** `pytest tests/unit/test_memory/test_consolidator.py -n0` green; `ruff format --check` + `ruff check` clean.
- [ ] **A8 — `<promise>M3 BATCHED</promise>`**, push, PR.

**Fixed-point / completion:** A2 red observed → A5 green → codex ACCEPT → ruff clean.

---

## Shard B — L-code-batch (L1, L3, L5, L6, L7, L14, L18)

Each L-item is its own red→green→commit micro-cycle in one worktree/PR. **L2 and L10 are conditional:** write their red test first; if L2 needs a server-side atomic RPC (migration) it moves to Shard D; if L10's short-circuit is provably unreachable (dead code, not a bug) it becomes a documented no-op or drop, not a "fix."

**L1 — episodic filters ignored.** `episodic_memory.py`: `min_importance`/`days_back` declared but never forwarded to `filter_params` in `search_episodic_memory()`.
- [ ] Red `test_episodic_search_respects_min_importance_and_days_back` (assert forwarded params reach the RPC arg dict). Run red → implement forwarding → green → commit.

**L3 — unvalidated `block_ms`.** `coordination/signals.py:135`: `iter_messages(block_ms=...)` accepts `0`.
- [ ] Red `test_signal_consumer_rejects_nonpositive_block_ms` (expect `ValueError` on `block_ms<=0`). Red → add validation → green → commit.

**L5 — session uuid mismatch.** `cognitive_integration.py:322,331`: generated `session_id` not passed to `working_memory.create_session()`.
- [ ] Red `test_cognitive_session_uuid_continuity` (same `session_id` across two `process_query` calls maps to one session). Red → pass `session_id` through → green → commit.

**L6 — naive timestamp.** `cognitive_integration.py:777`: `str(datetime.now())`.
- [ ] Red `test_graphiti_timestamp_is_tz_aware` (assert ISO-8601 with UTC offset). Red → `datetime.now(timezone.utc).isoformat()` → green → commit.

**L7 — crystallizer pagination cap.** `crystallization/crystallizer.py`: candidate SELECT lacks `.range()`, silently truncated at PostgREST 1000-row cap.
- [ ] Red `test_crystallizer_paginates_past_1000` (1100 candidates all processed). Red → `.range()` loop until exhausted → green → commit.

**L14 — singleton init race.** `services/factories.py`: check-then-set without a lock; concurrent asyncio startup can create duplicate clients.
- [ ] Red `test_singleton_client_concurrent_init` (10 concurrent `asyncio.gather` calls → `create_client` invoked once). Red → guard init with an `asyncio.Lock` (double-checked) → green → commit.

**L18 — single-match extractor.** `extractors/e2i_extractor.py:197-234`: `str.find()` finds only the first relationship per pattern.
- [ ] Red `test_extractor_finds_all_relationship_matches` (text with 2 matches → 2 extracted). Red → `re.finditer` → green → commit.

**B-final:**
- [ ] **codex-rescue** on the whole batch diff (anti-mocking paragraph included) → ACCEPT.
- [ ] `pytest tests/unit/test_memory/ -n0` (targeted) green; ruff clean.
- [ ] `<promise>L-BATCH GREEN</promise>`, push, PR.

---

## Shard C — H1: brand-scope the causal graph

**Why:** `query_semantic_paths` (`memory.py:706`) is **fail-closed to cross-brand admins** because FalkorDB causal-graph nodes carry no `brand` property (per the in-code comment, verified on live `e2i_causal`). H1 = (1) add `brand` to node creation, (2) brand-scope `traverse_causal_chain`/`find_causal_paths_for_kpi`, (3) relax the gate so a scoped viewer sees only their brand's paths.

**⚠️ Security-sensitive + faithfulness caveat:** this relaxes a deliberate BOLA fail-close (#690). There is **no branded graph data today**, so a faithful test must (a) build a small branded test graph in FalkorDB and (b) assert a scoped viewer gets *only* their brand and *cannot* see another brand's chains. This shard gets **extra codex-rescue scrutiny + my hands-on review before the promise fires** (ralph-loop alone is not trusted to relax a security boundary).

**Files:** `src/memory/semantic_memory.py`, `src/api/routes/memory.py`, tests.

- [ ] **C1 — Red (isolation):** `test_query_semantic_paths_scopes_to_caller_brand` — seed a branded FalkorDB test graph (BrandA + BrandB paths); a BrandA-only viewer query returns only BrandA paths; assert **zero** BrandB leakage. Fails today (no brand on nodes / admin-only gate).
- [ ] **C2 — Red (gate relaxation):** `test_query_semantic_paths_allows_scoped_viewer` — non-admin viewer with a brand grant gets 200 (not 403) and brand-filtered results.
- [ ] **C3 — Run red** → both FAIL.
- [ ] **C4 — Implement node `brand`:** add `brand` to CausalPath/KPI/Patient/HCP node creation (MERGE/CREATE) in `semantic_memory.py`; backfill strategy noted (existing unbranded nodes remain admin-only / excluded from scoped results — fail-closed for un-branded).
- [ ] **C5 — Implement query scoping:** add a `brand`/`brands` filter to `traverse_causal_chain` + `find_causal_paths_for_kpi` Cypher (`WHERE n.brand IN $brands`); thread the caller's grants (`get_user_brands`) from the route.
- [ ] **C6 — Relax the gate** in `memory.py:706`: replace the `is_cross_brand_admin` fail-close with brand-scoped access; admins still see all; un-branded nodes stay admin-only (never leak to a scoped viewer).
- [ ] **C7 — Run green** → both PASS; add an explicit cross-brand-leak negative test and confirm it stays red→green correctly.
- [ ] **C8 — codex-rescue** (security-focused brief: "prove a scoped viewer cannot read another brand's paths; flag any path where un-branded nodes leak") → ACCEPT.
- [ ] **C9 — my manual review** of the diff + the leak test before the promise.
- [ ] `pytest` targeted green; ruff clean. `<promise>H1 BRAND-SCOPED</promise>`, push, PR.

---

## Shard D (separate / deferrable) — L20 + L22 migrations
- **L22:** `CREATE OR REPLACE sync_hcp_patient_relationships_to_cache` accumulating the 2nd-block ROW_COUNT (`001b:299,321`). Red test asserts returned count == both blocks. New `database/memory/0NN_*.sql`.
- **L20:** drop redundant IVFFlat on `episodic_memories.embedding` (HNSW in `011` covers it). **Gate:** faithful prod `EXPLAIN ANALYZE` parity check first (read, needs auth); migration only after parity confirmed. Otherwise defer.
- Out-of-band application (like 030/031/032/033/034); flag for manual apply.

---

## L21 closeout (no build)
- [ ] (optional, needs auth) read-only droplet check that `verify_insight_chain` has the `ml_prediction` branch (030 applied).
- [ ] Tick L21 done in the #694 ledger with a note pointing at `030` / `d39fb7d5`.

---

## Execution mechanics

**Order (low-risk first to validate the loop machinery): A (M3) → B (L-batch) → C (H1) → D (optional).**

**Per shard:**
1. `git worktree add -b fix/mem-<shard> <path> origin/main` (fresh, disjoint from PR #763).
2. Run the **ralph-loop** with a shard-specific prompt that encodes: "TDD red-first per the plan; run tests `-n0`; on green, run `codex:codex-rescue`; only emit `<promise>…</promise>` after codex ACCEPT + ruff clean." Use `--max-iterations` as a backstop and the per-shard completion promise.
3. `codex:codex-rescue` brief **must** include verbatim: *"If a recommendation solves a labeling problem instead of a functional problem, flag it HIGH. If it preserves/【deletes】code without investigating intent, flag it HIGH. Audit the question, not just the answer."*
4. On promise: ruff (`format --check` + `check`), targeted pytest `-n0`, push (proxy bypass set), `gh pr create` (target #694).
5. CI green → merge-commit (never squash); `--admin` only if BLOCKED purely on the review gate. **No deploy** (deploy.yml stays HELD).
6. Clean up worktree.

**Droplet discipline:** pytest `-n0` (xdist workers OOM under memory pressure); never whole-tree mypy (CI arbiter); targeted runs only.

---

## Open questions for approval
1. **PR granularity:** 3 PRs (A, B, C) + optional D — OK? Or fold L21-closeout/ledger tick into B?
2. **H1 autonomy:** OK to relax the #690 fail-closed BOLA boundary now (with the branded-graph test proving no cross-brand leak), or keep H1 admin-only and defer until branded graph data exists in prod?
3. **L2/L10:** acceptable to let the red-first test decide (L2→Shard D if it needs an RPC migration; L10→drop if provably unreachable)?
4. **L20:** include the IVFFlat drop now (with a prod perf-parity check) or defer to a perf window?
5. **Deploy:** confirm everything stays HELD (no deploy) — these are merges only; memory migrations (D) applied out-of-band on your go.
