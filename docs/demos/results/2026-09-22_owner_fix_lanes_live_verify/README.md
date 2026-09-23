# Owner-fix lanes on PRs #2223 + #2224 — merge, deploy, live verification (2026-09-22/23)

Owner directive (2026-09-22 ~16:40Z) on the four decisions the two lane PRs had surfaced:
cohort contract → fix; Feast gate/beats/queue → investigate and fix; 0.7 prior + no unique key → fix;
fidelity chain "skipped" → surface proposed experiments. Then (21:45Z): "merge, deploy, live-verify".

## What merged and what shipped

| PR | merge commit | merged (UTC) | deploy run | deployed content (`docker inspect`) |
|---|---|---|---|---|
| #2223 `claude/lane-2207-ml-persistence-wiring` (17 lane commits + 2 merges, head `70af909b7`) | `5d54caa8c` | 21:50:26 | 35788998682 → success | converged to main HEAD `c4afb04cf` (#2223 + #2231) |
| #2224 `claude/lane-2206-twin-fidelity-honesty` (7 lane commits + main merge, head `691998497`) | `716682d34` | 22:26:35 | 35792404834 → success | converged to main HEAD `814eaa959` (#2224 + Lane B #2230) |
| Lane B #2230 (other session) | `814eaa959` | 23:13:45 | 35796394678 → success | same-sha redeploy of `814eaa959` |

Both PRs: every CI workflow green on the merged head, mypy report identical to main (58 == 58 after
line-number normalisation), merged with `--merge` (never squash). #2206 auto-closed; #2207 closed by
hand (its "Closes #2207" line was not auto-linked).

Deploy ledger (`docs/demos/results/2026-09-15_trx_canonical/rollback/attempts/`): `pr2223`, `pr2224`,
`lane_b_814eaa959` recorded from the REST job logs; `prove_state.sh new` → **75 OK / 0 BAD**
(`prove_state_new_final.txt`). Migrations applied by the deploys: `150_ml_registry_cohort_contract.sql`
(22:37Z) and `ml/046_ab_results_one_final_per_experiment.sql`. Both Feast sidecars rebuilt and healthy;
`e2i_feast` logs `starting feast serve (materialize endpoints registry-locked)` (serve_locked.py live).

## Live verification

`live_verify.py` (read-only): **29/29 PASS** on the final stack (`live_verify_readonly_final.txt`) —
container images == origin/main; sidecars healthy + reachable from worker_medium; migrations in the
ledger; the three `cohort_*` columns present and NULL on all 14 real registry rows (no backfill); the
partial unique index present with 360/360 final rows; 30 twin simulations, 0 linked; Celery routes
(`fidelity_tracking_update`, `execute_model_retraining`, the Feast beats) all on `analytics` and
consumed; `FEAST_URL` set; admin JWT; `/digital-twin/models` 3 rows `unvalidated` / `synthetic_target`
/ `shared_fit_model_count 3`; `/digital-twin/proposed-experiments` → `total_proposed 30, total_linked 0,
real_experiments_running 0, outcome_measurable_in_real_mode false`, every item an unlinked
deploy/refine proposal with n + weeks + `proposal_basis twin_simulation`; simulation list carries
`experiment_design_id`; OpenAPI lists the proposals + draft routes and the retraining trigger.

`live_verify.py --exercise-beats` (production actions, the scheduler's own calls;
`live_verify_exercise_beats.txt`): **31/31 PASS**.
- `materialize_incremental_features` → "Feast client initialized in remote mode: http://feast:6566" →
  `POST /materialize-incremental` 200 on the sidecar ("Materializing 10 feature views … into the redis
  online store") → **10 `ml_feast_materialization_jobs` rows, status `success`** (table was at 0 rows).
- `check_feature_freshness` → 10 `ml_feast_feature_freshness` rows, all `stale` — the real #559 recency
  (source tables last written ≥ 41 h ago); the task logs the ALERT and reports `fresh: False` honestly.
- `check_retraining_for_all_models` → fanned out one `evaluate_retraining_need` per real model on
  worker_light; decisions flagged `should_retrain: True` (performance_degradation) end with
  `retraining_triggered: False, requires_approval: True` — held by the approval gate
  (`auto_approve=False`) before the cohort guard; `ml_retraining_history` stays at 0 rows.

Stale `ml`-queue messages: still **15** on broker db 1 (owner action, permission-gated for the agent).

## Files
- `live_verify.py` — the checks (read-only by default; `--exercise-beats` enqueues the three tasks).
- `live_verify_readonly.txt` (first run, before the Lane B same-sha redeploy; 27/29 — the two FAILs were
  the script probing `/openapi.json` instead of `/api/openapi.json`, fixed), `live_verify_readonly_final.txt`,
  `live_verify_exercise_beats.txt`, `prove_state_new_post_pr2224.txt` (2 BAD: a newer deploy run existed),
  `prove_state_new_final.txt` (75/0).
