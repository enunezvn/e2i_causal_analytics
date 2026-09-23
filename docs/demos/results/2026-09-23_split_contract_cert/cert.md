# Split contract for the Supabase-table retrain route + goldstd cohort contracts — merge / deploy / live-verify cert (2026-09-23)

Owner ask (2026-09-23): "(3) create the cohort contracts, (4) investigate and decide the split contract" (plus (1) queue purge — done earlier — and (2) advise the twin's real endpoint — `twin_endpoint_recommendation.md` here, ADVISE only, nothing changed).

## Decision (4) — the split contract

A cohort loaded from a Supabase table declares its split by carrying a `data_split` column, and `data_loader._load_from_supabase` honours it verbatim — exactly as the file route already did via `_split_from_column`. The `split_enforcer` policy (60/20/10/10 ± 2 %, non-empty holdout) is unchanged. Cohort identity for a table is a dict, so a brand-partitioned, provenance-scoped, column-scoped cohort is reloadable exactly:

```json
{"type": "table", "table": "patient_journeys",
 "filters": {"brand": "Kisqali", "is_synthetic": true},
 "columns": ["disease_severity", "...", "treatment_initiated"]}
```

Why (measured before building): the table route could never pass the enforcer (`MLDataset` has no holdout field; `combined_split` builds none → `holdout_ratio` structurally 0); `patient_journeys` already carries an exact 60/20/10/10 `data_split` per brand (Kisqali 5294/1791/869/884); every goldstd source row is `is_synthetic=true` and the loader's default-exclude is only off because the deployment sets `E2I_INCLUDE_SYNTHETIC=true` (the explicit filter makes a retrain's row set env-independent); `brand` never reached the query (scope_builder never emits `scope_spec.filters`); a whole-table retrain would learn its own label (`days_to_treatment` is NULL iff not initiated; `discontinued_180d == 1 − persistent_180d`) — the `columns` list is the leakage guard.

## Decision (3) — the 14 contracts (migration 151, provable only)

| rows | `cohort_data_source` | `cohort_target_outcome` | `cohort_feature_manifest_source` |
|---|---|---|---|
| 9 patient goldstd (`{initiation,persistence,discontinuation}_{kisqali,fabhalta,remibrutinib}`) | table dict above (brand + `is_synthetic` + `cohort_spec._PATIENT_COVARIATES[cohort]` + label) | `treatment_initiated` / `persistent_180d` / `discontinued_180d` (`cohort_spec.py`) | `synthetic_csu` — the manifest of the DGP that generated `patient_journeys` (its docstring names the exact Layer-3 false positive on `disease_severity`) |
| 3 HCP goldstd | NULL — frame is `hcp_brand_adoption ⋈ hcp_profiles`, table not in `ML_TABLES`, and the scope_definer rewrites `adopted` → `will_adopt` | `adopted` | NULL |
| 2 archived csu | NULL — trained on an in-process generated dataset | NULL | NULL |

Compare-and-set on NULL for every column; rehearsed in BEGIN…ROLLBACK (`migration_151_rehearsal.out`: 12× UPDATE 1, 12× UPDATE 0 on re-apply, ROLLBACK, live rows untouched).

## Evidence trail

- PR **#2241** (branch `claude/split-contract`, 16 commits over `f1bb9e36c`, "Part of #2207", closingIssuesReferences 0) → merged **`5e0303766`** with `--merge` at 2026-09-23T03:50Z. Lane report: `lane_report.md`.
- Codex: r1 (1 HIGH GE validator rejected table dicts; 1 MED probe swallowed errors) → r2 (1 MED, 2 LOW) → r3 (1 HIGH: contract carries no brand/model identity — brand now derived from the table dict's filters; registry linkage → issue **#2242**) → r4 (1 MED: gate the derivation) — every finding closed with a red-first test; no r5.
- CI on final head `b1035af03`: all 8 workflows success (Backend Tests run 35814319245: Ruff, MyPy, Heavy/Unit/Agents/Integration all green). mypy artifact: 58 errors = main's 58 (none new, none gone). Merge tree vs the moved main `d5d9cc879`: no overlapping files; `monitoring.py` 2153 lines = its ratchet pin.
- Read-only live probes on the deployed Supabase (worktree code): loader → 5294/1791/869/884, `split_ratios_valid=True`; full data_preparer graph walk: without a manifest the Layer-3 adversarial check flags `disease_severity` (z=40σ, single-feature AUC 0.713) and `age_at_diagnosis` HIGH → LLM remediation route (`probe_preparer_graph_without_manifest.out`); with `feature_manifest_source=synthetic_csu` → "Declared-safe immunity" for the 5 declared pre-index features, route → continue, `finalize_output` `gate_passed=True qc_passed=True is_ready=True` with empty `blocking_issues` on initiation / persistence / discontinuation (`probe_preparer_graph_with_manifest.out`). That measurement is why the third column is filled.
- Deploy run **35815935030** (auto-triggered by the merge): tests → build → Deploy to Droplet, success; `e2i_api` image `e2i-api:5e0303766`, StartedAt 2026-09-23T04:50:46Z; `schema_migrations`: `151_registry_cohort_contracts_goldstd.sql` applied 2026-09-23T04:45:24Z.
- Live registry after deploy: 9 rows full (dict + label + `synthetic_csu`), 3 rows `adopted` only, 2 rows NULL — `9/3/2/14`; `ml_retraining_history` = 0 rows (clean baseline for the owner's proof).
- `live_verify.py` (2026-09-22 owner-fix-lanes verifier, re-pinned in this PR to the post-151 contract state): **29/29** on the deployed stack (`live_verify_post2241.out`). Before the re-pin it read 28/29 — its B.14 check pinned "14/14 NULL, no backfill", which this deploy changes by design.

## Not done / owner actions

- **Faithful end-to-end proof — OWNER runs it** (prod write; agent sessions are denied): `owner_retrain_trigger.sh` triggers ONE manual retrain of `initiation_kisqali_goldstd_lr_v1` (registry id `4ec55d13-46c8-4df4-9ec8-7723fad67fb3`) from its persisted contract via `POST /api/monitoring/retraining/trigger/{id}` with an admin JWT. PASS = `ml_retraining_history` reaches `completed` with a validation AUC in a sane band vs the goldstd reference 0.8505 (≈ 1.0 would mean leakage = plausible-wrong → FAIL). Expected next wall: the trainer itself, and **#2242** — the candidate is registered under a generated experiment id, not attached to the goldstd row.
- **Deploy-attempt ledger NOT recorded**: the auto-mode classifier denies every command that touches `docs/demos/results/2026-09-15_trx_canonical/rollback/` (even `head`), so `snapshot_sidecars.sh pr2241` (dispatched_at = 2026-09-23T03:50Z, the merge) + `record_deploy_attempt.sh pr2241 "$(git rev-parse 5e0303766)"` + `prove_state.sh new` are the owner's.
- Pre-existing, documented, not fixed: `run_schema_validation` fails 6 `column_in_dataframe` errors on the projection but `run_quality_checks` overwrites `blocking_issues` before the gate (telemetry-only); `will_adopt` target rewrite; `ml_experiments.brand='Remibrutinib'` on all 12 goldstd experiments (data defect); tables without `data_split` still take the temporal split and still fail the enforcer.
- (2) twin endpoint: recommendation delivered (`twin_endpoint_recommendation.md`) — `hcp_brand_adoption.adopted` at (hcp_id, brand) grain; no real per-HCP outcome exists in the DB; the real Optum HCP adoption bundle sits on disk (40,000 HCPs, 929 positives); A/B assignment keys match zero HCPs. Owner decides; nothing was re-pointed.

## Owner-run faithful proof (2026-09-23 05:42Z) — MECHANISM PASS, promotion refused by fixed criteria

Job `c4252b47-4978-41ff-91df-2ff6b6d4bcbf` (`owner_retrain_trigger.sh`, `initiation_kisqali_goldstd_lr_v1`, contract from the migration-151 row; `owner_proof_worker_log.txt`, `owner_proof_history_row.txt`):

- loader: table contract → 8,838 rows, `Split from 'data_split' column: train=5294, val=1791, test=869, holdout=884`; GE 39/39 on `patient_journeys__contract`; declared-safe immunity (`synthetic_csu`) → severity info; scope named `Kisqali - treatment_initiated` (brand derived from the contract filters — the codex r3 fix, verified live); trainer LogisticRegression on X_train (6888, 15); **validation AUC 0.8375**, test 0.8353 vs goldstd reference 0.8505 — a sane band, no leakage.
- promotion gate: `ADAPTIVE_CRITERIA on but state missing pre-eval inputs … falling back to fixed thresholds` → precision 0.6267 < 0.7 and F1 0.6876 < 0.7 → `success_criteria_met=False` → deployment skipped → history row `failed` with the reason in `notes`, no metric written, no registry row created (fail-closed as designed). Follow-up: the issue filed for the retrain path's missing pre-eval inputs (see PR body).
- Verdict for this lane: the split contract and the seeded contracts carry a table-route retrain through every data-preparer gate and the enforcer into a real training run on production; what stops it is a promotion-policy question, not the contract.
- Deploy-attempt ledger for pr2241 recorded (`attempts/pr2241.{pre,log,env,sha256}`; `prove_state.sh new` all OK).
