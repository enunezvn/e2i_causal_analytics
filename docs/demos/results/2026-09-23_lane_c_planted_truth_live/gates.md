# Lane C-load — pre-load gates (2026-09-23, worktree `.worktrees/lane-c-load` on main e0468f0de; deployed e2i_api = 814eaa959, started 2026-09-23T00:15:46Z)

All pytest runs `-n 0 -p no:cacheprovider --timeout=600` under the box lock with dead Supabase / Redis / MLflow env.

| gate | result |
|---|---|
| RED (before `scripts/load_csu_escalation_cohort.py` existed) | `tests/unit/test_scripts/test_load_csu_escalation_cohort.py`: 1 error during collection (module absent) |
| GREEN r0 (engine + CSU entrypoint) | 69 passed, 1 skipped (`test_optum_causal_cohort_realdb.py`, `E2I_DB_INTEGRATION` gate) — CSU file + the UNCHANGED Optum test file + the contract test |
| Teeth r0 (plant: CSU provenance rule never refuses; guard bypassed) | 5 failed / 56 passed — exactly `test_load_frame_fails_loud[is_synthetic0,1]`, `test_specs_refuse_each_others_rows`, `test_main_execute_refuses_when_the_dataset_guard_is_absent`, `test_main_execute_refuses_when_the_planted_truth_seam_is_open`; every Optum test green under the plant; engine restored byte-identical (cmp) |
| codex r1 (read-only, branch-wide, pushback paragraph) | `VERDICT: REVISE` — 6 HIGH, 2 MED, 1 LOW; all fixed red-first (below) |
| GREEN r1 (after the codex fixes) | 83 passed, 1 skipped |
| Teeth r1 (4 plants: non-bool provenance accepted; `upsert` boundary unguarded; deployed attestation bypassed; reversed page order) | 9 failed / 66 passed — the 3 non-bool provenance variants, both `upsert` boundary tests, both deployed-guard tests, the ordered-paging test, the deployed-refusal `main` test; Optum green; engine restored byte-identical |
| ruff (CI's two commands, whole tree, `--no-cache`) | All checks passed! / 3438 files already formatted |
| debris grep over the change set (`if False`, `SIMULATED`, `e2i-api:<40-hex>`) | empty |
| dry run vs PROD (read-only; `preload_dry_run.txt`) | WOULD WRITE n=3000 (RHAPSIDO 1274 / XOLAIR 1220 / DUPIXENT 506; treatment 1=1274, 0=1726); PROVENANCE every row true; GUARD clear (tree guard covers the dataset, seam closed, deployed 814eaa959 descends from guard commit c0860bbf4); LIVE BEFORE n=0, provenance {true: 0, false: 0}; only GET requests in the HTTP log |
| backing artefact | `data/rwd/synthetic_CSU/csu_escalation_causal/` built twice → identical sha256 (see `owner_load_command.md`) |

## codex r1 findings → fixes

| sev | finding | fix |
|---|---|---|
| HIGH | truthiness (`astype(bool)`) accepts int/string provenance — `"false"` reads True | `provenance_problem()`: every value must be exactly `bool`/`np.bool_`; NULL / int / string refused; shared by `load_frame` and `upsert`; tests for `"true"`, `"false"`, `1`, a trailing NULL, and the nullable `boolean` dtype (accepted) |
| HIGH | public `upsert` bypassed provenance + guard | enforced at the write boundary before the first batch (ValueError / RuntimeError, nothing written); tests |
| HIGH | guard checked the local tree, not the deployed instance | `deployed_guard_problem`: the `e2i_api` image commit (docker; or `--deployed-commit <sha>`) must descend from the #2228 merge commit c0860bbf4 (`git merge-base --is-ancestor`); unreadable → refused; tests with seams + a real tag-parser test |
| HIGH | `run_e2e_live.py` missed the independent `validation_outcomes` writer (`log_validation_outcome_with_status`, refutation.py:1744) | replaced by a recorder returning a non-durable `StoreResult`; call count recorded and asserted ≥ 1 |
| HIGH | the deployed probe was a state-changing POST | now `GET /causal/brands?dataset=csu_escalation_causal` (same dataset predicate, read-only), expected `[]` |
| HIGH | import-time upward dotenv discovery | `load_env()` runs only on the CLI path (`run`) and in `get_client()`; a test pins no top-level dotenv call |
| MED | `.range()` paging without an order | `.order(patient_id)` before `.range()`; both fakes implement `order`; a 1,500-row two-page test |
| MED | Opik / DSPy signal routing not explicitly neutralised in the live script | `OPIK_ENABLED=false`; `route_causal_impact_signal` → recorder; audit initializer verified store-less |
| LOW | Optum messages not byte-identical (one-arm wording, NULL hardening) | documented in the Optum module docstring; the Optum test file's assertions were not changed (only `order()` added to its fake) |

## Not done
- No `--execute` (prod write): the owner runs it (`owner_load_command.md`).
- `run_e2e_live.py` NOT run yet — it needs the loaded table; run after the owner's load, then `cert.md`.
