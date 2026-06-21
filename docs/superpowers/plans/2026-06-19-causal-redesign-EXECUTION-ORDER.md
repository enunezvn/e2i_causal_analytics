# Causal Redesign — Execution Order & Cross-Plan Contract

> **MANDATORY: read this before executing ANY phase plan.** The five phase plans (P1, P1b, P0, P2, P3, enrichment) were drafted in parallel and self-reviewed in isolation. A consistency pass found 3 blocking conflicts + drift/sequencing issues. This doc is the authoritative reconciliation; where it conflicts with a phase plan, **this doc wins**. When executing subagent-driven, inject this doc into every task subagent's context.

## Verified facts (settle the cross-plan assumptions)

- **D1 ✓** `summary` IS a field on `DiscoveredEffect` in #1030 (`schemas/causal.py:694`, worktree `wt_causal_discovery_revamp`). P0/P2 may consume `e.summary` once #1030 merges. P1 does NOT add `summary` (only `brand` + `adjustment_set`).
- **Q2 ✓** `batch_loader.py:838` (`available_columns = TABLE_COLUMNS.get(table_name, ...)`) selects only known columns → a `grain` tag on generator rows is **stripped before insert** (no 400). `causal_paths` has no `grain` column → runtime grain-scoping is by **spec membership**, not a column.
- **Q3 ✓** `hcp_brand_adoption.is_synthetic` exists (probed) → P2's `_te_paged_select` reuse (which filters `is_synthetic`) returns rows.

## Execution order (dependency DAG)

```
#1030 (merge first; gated)
   ├─► P1  (patient backend; #1030-independent, may start in parallel with the #1030 merge)
   │      ├─► P1b (geo one-hot + agent priors)
   │      ├─► P2  (HCP grain)  ──► P3 (trigger grain)     [generator edits sequenced P1→P2→P3]
   │      └─► P0  (unified FE page; also needs #1030 FE)
   └────────────────────────────────────► enrichment (Clinical Context; needs #1030 + **P0**)
```

Authoritative sequence: **#1030 → P1 → {P1b, (P2 → P3), P0} → enrichment.** P1 may begin before #1030 lands (it touches no #1030 files); everything else has #1030 as a checked precondition (**S1**).

## Resolutions (apply these; they override the phase plans)

- **B2 ✓ FIXED in P1** — `GeneratorConfig` imports from `src.ml.synthetic.generators.base`; generator tests **append** to the existing `test_causal_paths_generator.py`.
- **D2 ✓ FIXED in P1** — the grain-scope guard (`if t not in spec["treatment"] or o not in spec["outcome"]: continue`) now lives in P1's `_discover_candidate_questions`. **P2/P3 must NOT re-add it** — they only add their dataset to `_CAUSAL_DATASET_SPECS` and the guard scopes automatically.
- **B3 — single `grain` convention.** Every generator row carries `"grain"` ∈ {`"patient"`,`"hcp"`,`"trigger"`} (stripped at load per Q2). P1 tags patient rows `"patient"`; P2 tags HCP rows `"hcp"`; P3 tags trigger rows `"trigger"`. **All shared-test rescopes use the single predicate `df[df["grain"] == "<g>"]`** (not `end_node != "adopted"`). Generator edits are appended in order **P1 → P2 → P3**, each before `return pd.DataFrame(rows)`; P2 and P3 rebase onto the merged generator.
- **D3 — ONE coercion helper, owned by P1.** P1 extracts the per-row coercion loop in `_load_agent_estimation_frame` into `_coerce_estimation_row(row, *, treatment_var, outcome_var, numeric_cols, categorical_cols=frozenset(), derivations=None, fill_zero=frozenset()) -> Optional[dict]` (returns None when a treatment/outcome value is missing). The three extenders then pass extra args, never re-rewrite the loop:
  - **P1b** adds `categorical_cols` (pass-through, no float-coerce) + one-hot expands geo dummies on the assembled DataFrame *after* the loop, and rewrites `select_cols` accordingly.
  - **P3** passes `derivations` (`acceptance_status`/`action_taken` → 0/1) + `fill_zero` (`conversion_flag` NULL→0).
  - **P2**'s JOIN loader calls `_coerce_estimation_row` per merged row (after deriving `centrality_z`) instead of duplicating the loop.
- **D5 — geo dummies must reach the LEADERBOARD path too.** P1b must also update `_run_discover_effects_task`: after `df, select_cols = await _load_agent_estimation_frame(...)`, thread the **expanded** `select_cols` (minus treatment/outcome) into the per-question agent request/initial-state — not just `run_causal_agent_analysis`. Otherwise leaderboard retention rows silently skip the geo confounder. Add this as an explicit P1b step.
- **B1 — enrichment FE re-target.** P0 deletes `frontend/src/pages/CausalDiscovery.tsx`. Enrichment Task 8 must mount `<ClinicalContextPanel>` in **`frontend/src/components/causal/CausalAnalysisDetail.tsx`** (after its `estimator_comparison` block) and the leaderboard MoA chip in P0's `CausalAnalysis.tsx`. Add **enrichment → P0** as a hard dependency. `AgentCausalAnalysisResponse` has no `brand`, so P0 must thread `brand` into `CausalAnalysisDetail` as a prop for `useClinicalContext(brand, outcome)`.
- **S3 — ONE gated reseed, replace-not-append.** The generator mints fresh `path_id`s each run (`scp_{uuid4}`) and the loader upserts by `path_id` → re-running ACCUMULATES synthetic rows. So: run the reseed **once**, from the final merged generator (after P3), and **DELETE `causal_paths WHERE is_synthetic IS TRUE` before loading** (snapshot first). P1/P2 reseed steps become "verify-only" if run earlier. All reseeds stay gated on explicit user authorization.
- **Q1 — enrichment service snippet bug.** In enrichment Task 5 `service.py`, the `__init__` default must be `self._mechanism = mechanism_provider or _default_chembl()` (NOT `... or ChEMBLMechanismProvider(client=_default_chembl())`, which double-wraps). Use the corrected line in the code block; delete the corrective prose note.
- **S2 — P0 dependency note** should read "needs `brand`+`adjustment_set` (P1) and `summary`+`estimator_comparison` (#1030)."

## Pre-execution checklist (per phase)

- [ ] #1030 merged to main (precondition for P1b/P2/P3/P0/enrichment; P1 exempt).
- [ ] Phase's hard deps merged (see DAG).
- [ ] Generator edits applied in P1→P2→P3 order with the shared `grain` tag.
- [ ] `_coerce_estimation_row` exists (P1) before P1b/P2/P3 extend it.
- [ ] Reseed run once, post-P3, delete-synthetic-first, gated on user OK.
- [ ] This doc injected into each task subagent's context.

## Clean plans (no cross-plan logic issues beyond the above)
- **P1** — keystone; now hosts the shared grain-guard (and will host `_coerce_estimation_row`).
- **Enrichment backend** — provider/service/clients design is self-consistent and correctly REST-not-MCP; only the FE host target (B1) + the one snippet (Q1) need the fixes above.
