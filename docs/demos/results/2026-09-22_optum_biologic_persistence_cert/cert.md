# Lane A live cert — `optum_biologic_persistence` loaded and estimated through the deployed API (2026-09-22)

**Verdict: PASS.** The real cohort (n = 15,209; XOLAIR 11,009 / DUPIXENT 4,200; `is_synthetic = false` on every row) is loaded into prod, the load is idempotent (measured), the real-DB probe passes, and the deployed `causal_impact` agent completed all four outcomes through `POST /causal/agent-analyze` inside the 900 s cap with discovery off and the curated DAG. The primary outcome `persistent_at_180d_g28` shows a small positive Dupixent effect (+3.3 pp, CI excluding 0, 4/5 refutations, gate proceed). **Two outcomes are null findings** (`discontinued_180d`, `biologic_switch_180d_flag`: CI contains 0) — reported as findings, not failures. The shipped `persistent_at_180d` row is the days-supply artefact and is not an effect. Nine findings are listed below; none blocks the default-path cert. Two are label/wording defects with PRs or follow-ups named, and F9 (the discovery-on live probe killed the API worker and orphaned its job) blocks any discovery-on run on this frame until fixed.

Plan: `docs/superpowers/plans/2026-09-22-lane-a-real-data-causal-pipeline.md` §Task 12. Spec: `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md` §5–§7. Pre-merge evidence (Tasks 7–9) is the rest of this directory.

## Deployed target (verified per step, not assumed)

| item | value |
|---|---|
| e2i_api image at cert time | `ghcr.io/enunezvn/e2i-api:c0860bbf4` (Lane C merged; the earlier 44d310f0f deploy at 20:23:25Z was superseded before the cert runs) |
| e2i_api `StartedAt` | `2026-09-22T21:22:09.275118978Z` — identical in `_cert.container_started_at` of all four raw responses (`grep` over `raw_*.json`: 4 × the same value) and unchanged after the last poll (22:11:57Z). A peer merge (#2223, main 5d54caa8c) started deploy run 35788998682 at 21:50Z; its container flip, if any, came after the four cert runs completed — the discovery probe's own StartedAt is recorded in `live_probes/cert.md` |
| migration ledger | `148_optum_biologic_persistence_causal.sql \| 2026-09-22 20:16:37.326868+00` (applied by the 44d310f0f deploy's runner); `149_csu_escalation_causal.sql \| 21:16:42+00` also present |
| `to_regclass('public.optum_biologic_persistence_causal')` | resolves |
| API base | `https://eznomics.site/api` |

## Step 2 — the owner-GO load (spec §7), rehearsed then executed, then re-executed

Dry-run (20:32 UTC, exit 0, `--dry-run` default):

```
WOULD WRITE: n=15209 arms={'XOLAIR': 11009, 'DUPIXENT': 4200}
LIVE BEFORE: n=0 arms={'XOLAIR': 0, 'DUPIXENT': 0}
DRY RUN complete. No rows written. Re-run with --execute to write.
```

`--execute` — run by the **owner** at 21:49 UTC (the cert agent's own attempt was denied by the session's permission classifier as a shared-resource write; that is the intended control for a prod write). Loader output, verbatim:

```
EXECUTE: upserted 15209 rows into optum_biologic_persistence_causal (idempotent on patient_id).
LIVE AFTER: n=15209 arms={'XOLAIR': 11009, 'DUPIXENT': 4200}
  persistent_at_180d_g28: positives={'XOLAIR': 8081, 'DUPIXENT': 3125} rate={'XOLAIR': 0.734, 'DUPIXENT': 0.744}
  discontinued_180d: positives={'XOLAIR': 1162, 'DUPIXENT': 511} rate={'XOLAIR': 0.1056, 'DUPIXENT': 0.1217}
  biologic_switch_180d_flag: positives={'XOLAIR': 106, 'DUPIXENT': 52} rate={'XOLAIR': 0.0096, 'DUPIXENT': 0.0124}
  persistent_at_180d: positives={'XOLAIR': 5759, 'DUPIXENT': 1458} rate={'XOLAIR': 0.5231, 'DUPIXENT': 0.3471}
ROW VERIFICATION: compared 15209 exported rows against 15209 live rows.
VERIFIED: live arm split, treatment counts, per-outcome positives, and every exported field of every row equal the parquet.
```

Independent re-read (psql, cert agent): `count 15209 | is_synthetic 0 | XOLAIR 11009 | DUPIXENT 4200`; `count(DISTINCT patient_id) = 15209`.

**Idempotence — measured.** The owner ran `--execute` a second time at 21:50 UTC: `LIVE BEFORE: n=15209 arms={'XOLAIR': 11009, 'DUPIXENT': 4200}` with per-outcome positives 8081/3125, 1162/511, 106/52, 5759/1458 → `EXECUTE: upserted 15209 rows` → `LIVE AFTER` identical to `LIVE BEFORE` on every line → `ROW VERIFICATION: compared 15209 exported rows against 15209 live rows.` → `VERIFIED`. Row count and distinct `patient_id` both still 15,209 afterwards.

## Step 3 — real-DB probe

```
E2I_DB_INTEGRATION=1 pytest -n 0 -p no:cacheprovider tests/integration/test_optum_causal_cohort_realdb.py -q -rs --timeout=300
1 passed, 15 warnings in 3.32s
```

Negative control for the probe itself: run against the still-empty table before the load it failed with `AssertionError: live table is empty — the owner-GO load has not run` (`assert 0 > 0`) — the gate has teeth.

## Step 4 — the four API runs (sequential, `run_cert.py`, discovery omitted from the request)

Estimator was **not forced**: Task 9 (`preflight.md`) showed FINAL-code Auto completing every outcome in 185–258 s; only the pre-fix runs had hit the 900 s cap. Request body per run: `{"treatment_var": "treatment_dupixent", "outcome_var": <outcome>, "dataset": "optum_biologic_persistence", "limit": 20000}`. Every submit returned the discovery warning **verbatim**:

> Structure discovery is OFF by default for dataset 'optum_biologic_persistence' (measured to fail on this frame; Lane D pending) — the curated common-cause DAG is used. Pass auto_discover=true to attempt it.

| outcome | n_rows | estimator (Auto chose) | ATE | 95 % CI | p | refutation passed/total | negative control | E-value / gate | dag_source | discovery_enabled | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`persistent_at_180d_g28`** (primary) | 15,209 | LinearDML | **+0.0325** | **[+0.0157, +0.0494]** | 1.50e-4 | 4/5 (bootstrap failed) | SKIPPED `no_negative_control_declared` (see F4) | 1.26 / proceed | domain_knowledge | false | 362.2 |
| `discontinued_180d` (secondary) | 15,209 | LinearDML | +0.0071 | [−0.0054, +0.0195] — **contains 0: NULL FINDING** | 0.266 | 3/5 (unobserved_common_cause, bootstrap failed) | SKIPPED `no_negative_control_declared` | 1.33 / proceed | domain_knowledge | false | 301.0 |
| `biologic_switch_180d_flag` | 15,209 | LinearDML | +0.0032 | [−0.0014, +0.0078] — **contains 0: NULL FINDING** | 0.169 | 3/5 (unobserved_common_cause, bootstrap failed) | SKIPPED `no_negative_control_declared` | 2.00 / proceed | domain_knowledge | false | 278.8 |
| `persistent_at_180d` (shipped) — **days-supply artefact — read with the grace sweep, not as an effect** | 15,209 | LinearDML | −0.1576 | [−0.1768, −0.1384] | 0.0 (as reported) | 5/5 | SKIPPED `no_negative_control_declared` | 2.22 / proceed | domain_knowledge | false | 286.3 |

All four: `status = completed`, `needs_review = false`, `requires_review = false`, DAG 63 nodes / 123 edges / one adjustment set of **61** covariates (the same k = 61 Task 9 measured after the loader prune), `adjustment_type = confounding`, `discovered_confounders = []`, `poll_cap_hit = false`. Analysis ids: g28 `b8858563-4798-4a7f-b496-c6beab3f9ea2`, discontinued `134a70f3-4229-4c87-aa00-d4b8ffa4d3e5`, switch `5ff83f39-9ebd-47e8-b5e8-582f31ac01a7`, shipped `4dc3a2f8-88ea-453a-9b69-d7516459af60`. Raw responses: `raw_<outcome>.json`; run log in the session scratchpad (`lane_a12_cert_run.log`, exit 0).

Naive (unadjusted) contrasts for the same runs, from the responses: g28 +0.0100 [−0.0056, +0.0256]; discontinued +0.0161 [+0.0047, +0.0275]; switch +0.0028 [−0.0011, +0.0066]; shipped −0.1760 [−0.1931, −0.1588]. On `discontinued_180d` the naive CI excludes 0 and the adjusted CI includes it — the 61-covariate adjustment removed the apparent raw excess.

The two null-finding responses carry the agent's own warning, verbatim (discontinued shown; switch identical with its CI): "No detectable effect at this sample size: The 95 % CI [-0.005, 0.019] includes zero at n = 15209. The estimate is reported as a null finding; no unmeasured confounder is needed to explain it."

Comparison with Task 9's in-process pre-flight (host, FINAL code): g28 +0.0336 [0.0168, 0.0504] 4/5 → live +0.0325 [0.0157, 0.0494] 4/5; discontinued +0.0050 [−0.0074, 0.0174] 4/5 → live +0.0071 [−0.0054, 0.0195] 3/5; switch +0.0034 [−0.0012, 0.0079] 4/5 → live +0.0032 [−0.0014, 0.0078] 3/5. Same sign, overlapping intervals, same estimator; the live runs lost one refutation each on the two null outcomes (F5) and ran 1.3–1.5× slower (F8).

## The two caveats, as data (`caveats.py` → `caveats.json`; raw drop joined to the 15,209 cohort ids, 0 unmatched)

`treatment_response` by arm — **brand-coupled, NOT used** as outcome or covariate:

| arm | controlled | indeterminate | uncontrolled | refractory |
|---|---|---|---|---|
| XOLAIR (11,009) | **4,109** | 5,535 | 1,259 | 106 |
| DUPIXENT (4,200) | **30** | 3,596 | 522 | 52 |

`max_consecutive_biologic_coverage_days` quartiles by arm — why the 60-day gap in the shipped `persistent_at_180d` is a dosing-interval artefact:

| arm | q25 | median | q75 |
|---|---|---|---|
| DUPIXENT | 14 | **14** | 18 |
| XOLAIR | 28 | **55** | 84 |

Spec §7 quotes the Xolair median as 45 d; **measured on the cohort it is 55 d** (F7). The Dupixent 14 d holds. The shipped column's −15.8 pp "effect" (row 4) is the same −17.6 pp raw gap the persistence-definition disproof showed collapsing to 3.5 pp under a 14-day grace and inverting from 28 d; row 1 (`_g28`) is the honest primary.

## Findings

- **F1 — two null CIs (finding, not failure).** `discontinued_180d` and `biologic_switch_180d_flag` both have 95 % CIs containing 0 at n = 15,209. The brand-robust secondary therefore does not corroborate the primary's +3.3 pp; the primary stands on its own with E-value 1.26 (a weak unmeasured confounder would explain it away).
- **F2 — `data_source` reads `"synthetic"` on all four responses although every row is real.** This showcase instance runs `E2I_INCLUDE_SYNTHETIC=true`; `serves_synthetic_rows()` follows the deployment flag, so the label says "synthetic" for a real-backed dataset (measured in-process: `docs/demos/results/2026-09-22_lane_c_live_verify/inprocess_guard_probe.txt` on PR #2231). **Not a data defect** — the loaded table has `is_synthetic = false` on all 15,209 rows (psql above). Fix: PR #2232 (real-backed Optum dataset keeps the real-mode predicate and the "database" label under the showcase flag), held unmerged so the container stayed stable for this cert.
- **F3 — the served `selection_reason` wording is misleading.** Every response says "Lowest energy score among: causal_forest=0.6684, linear_dml=0.6690, drlearner=0.6690, ols=0.6695 …" yet `selected_estimator = LinearDML`, not the lowest-scored `causal_forest`. The selection is correct by the documented rule (`EstimatorSelector`: candidates within `min_energy_score_gap = 0.05` form a tie band — here the spread is 0.0003–0.0009 — and within the band the fastest confounding-robust estimator wins, so LinearDML beats CausalForestDML), but the reason string names a rule that was not the deciding one. Wording follow-up; no effect on the estimates.
- **F4 — the negative-control SKIPPED verdict is not visible in the API response.** `refutation.tests` lists the five run tests only; `skipped_tests` exists in agent state (`src/agents/causal_impact/state.py`) but is neither serialized by `AgentCausalAnalysisResponse` nor persisted to any DB column (`information_schema` query: no `skipped`/`negative_control` column outside a validation view). The SKIPPED `no_negative_control_declared` in the table is therefore asserted from the in-process lookup (`_negative_control_outcome("optum_biologic_persistence", "treatment_dupixent", <each outcome>) → None` and `_default_auto_discover("optum_biologic_persistence") → False`, run in-process from this worktree at the deployed sha c0860bbf4 with `src.__file__` asserted) and the registry test `test_no_negative_control_is_declared_until_measured_on_this_source`, not read off the response. Follow-up: surface `skipped_tests` in the response.
- **F5 — refutation 3/5 live vs 4/5 pre-flight on the two null outcomes.** `unobserved_common_cause` failed live on both nulls (passed in Task 9); `bootstrap` failed on three of four outcomes (as in Task 9 for g28). The gate still says proceed on all four; `quality_tier = "poor"` (energy ≈ 0.67) with `requires_review = false` on all four (F6).
- **F6 — `quality_tier = "poor"` on every run** while `requires_review = false`. The tier reflects outcome fit (binary outcomes, energy ≈ 0.67), not causal validity; recorded so a reader does not mistake the proceed gate for a strong-fit claim.
- **F7 — spec vs measurement:** Xolair median coverage run 45 d (spec §7) vs 55 d measured; carry the measured value.
- **F8 — wall time:** live 279–362 s per outcome under gunicorn vs 185–258 s in-process pre-flight; all under the 900 s cap with ≥ 538 s headroom; poll cap 1000 s never hit.
- **F9 — discovery-on kills the worker and orphans the job (live probe).** With `auto_discover: true` the run reached gate AUGMENT and then blocked the event loop in the backdoor adjustment-set search on the augmented graph until gunicorn aborted the worker (`--timeout 120`, code 134); the job stays `running` with no failure surfaced. Neither the 900 s agent cap nor the PR #2203 failure path can fire on a blocked loop. Details and follow-ups in `live_probes/cert.md`. This is why spec §5 keeps discovery off for the real dataset; the default did its job.

## Live probe (lead's addition): `auto_discover: true` on the real dataset — FAILED TO COMPLETE (F9)

One run, `discontinued_180d`, guided discovery on (`live_probes/cert.md`, `live_probes/raw_discontinued_180d.json`, log excerpts alongside). Guided PC ran and the gate returned AUGMENT (74 discovered edges, 77.40 % confidence), then the worker was killed by gunicorn's 120 s heartbeat timeout (`code 134`) inside `_find_adjustment_sets → _satisfies_backdoor_criterion → networkx.is_d_separator` on the augmented graph, running on the event loop; the job is orphaned in `running`. None of the Lane D / Lane E / vote-rule fields reached a response. Container unchanged throughout; not re-run. **The discovery-off default for this dataset is what let the four runs above complete.**

## What this cert does NOT claim

- No negative-control outcome was measured; SKIPPED is the honest verdict (spec §5).
- The primary effect is small and E-value 1.26; the secondary is null. This is an estimate under the curated common-cause DAG with 61 adjusters, not a clinical conclusion.
- The shipped `persistent_at_180d` row is carried only next to the grace sweep.
- Nothing here was projected; every number above is read from a response, a log, or a psql re-read.
