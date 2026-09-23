# Lane C planted-truth verification on the LIVE `csu_escalation_causal` table (2026-09-23)

**PASS.** After the owner's load (3,000 synthetic rows, `VERIFIED`, exit 0), the production loader read the live table through the dataset-level provenance guard and the REAL causal_impact graph recovered the planted ATE: **0.4071 vs true 0.364, error 0.0431** — inside the PR's `RECOVERY_TOLERANCE` 0.06 AND below half the naive error (0.0668), inside the 0.10 sidecar tolerance. Real mode answered **503** before and after (the planted rows are not served as real), the deployed container's brand dropdown answered **`[]`** for the dataset although the table now carries three brand labels, and every write seam was a recorder (counts below). 22/22 checks true.

| item | value |
|---|---|
| run | `run_e2e_live.py --api-probe`, in-process from the lane worktree at `a95c40e46` (PR #2239), real `.env`, `E2I_INCLUDE_SYNTHETIC=true` in the process exactly as the container carries it; 05:50:28Z → 05:51:59Z |
| container BEFORE / AFTER | image commit `5e03037668b6fdb0179c935f6650f7454661f8f2` (the #2246 deploy), `StartedAt 2026-09-23T04:50:46Z` — identical before and after (`container_before.txt`, `container_after.txt`): no restart mid-run; `5e03037` descends from the #2228 guard merge `c0860bbf4` |
| live table (read-only, `fetch_live_split` / provenance counts) | n=3000; arms XOLAIR 1220 / DUPIXENT 506 / RHAPSIDO 1274; treatment 0=1726 / 1=1274; `is_synthetic` true 3000 / false 0 — equal to the parquet's `WOULD WRITE` (`preload_dry_run.txt`) |
| loader `guard_problem()` | None (tree guard covers the dataset, seam closed, deployed image descends from the guard commit) |
| probe 1 — REAL MODE before (`PLANTED_TRUTH_RUN=False`, `deployment_includes_synthetic()=True`) | `serves_synthetic_rows=False`; loader → `HTTPException 503: No usable estimation rows … (treatment_remibrutinib -> persistent_at_180d_g28) in dataset 'csu_escalation_causal'` |
| probe 2 — PLANTED-TRUTH MODE (seam flipped in-process only) | submit: n_rows **3000 == live synthetic count**, `data_source=synthetic`, `auto_discover=False` (dataset default), k=13 resolved (age, charlson, elx_depression, lis_dual_flag, enrollment_duration_days, `payer_category=` ×3, `gdr_cd=M`, `geographic_region=` ×4 incl. `__missing__`), two arms |
| graph | `completed`, `dag_source=domain_knowledge`, estimator **LinearDML**, ATE **0.4071** [0.3734, 0.4407], p=0.0, naive 0.4976 (planted naive 0.4976), refutation 5/5 passed, 87.7 s, RSS 787 MB |
| recovery rule | error 0.0431 < 0.06 (`RECOVERY_TOLERANCE`) and < 0.0668 (0.5 × naive error 0.1336); < 0.10 sidecar; adjusted closer than naive |
| probe 3 — REAL MODE after | 503 again (the seam left no residue) |
| probe 4 — deployed API (read-only) | `GET /causal/brands?dataset=csu_escalation_causal` → HTTP 200 `{'brands': []}` |
| write seams (recorded, never written) | `validation_outcomes` writer 1 call (non-durable `StoreResult`), DSPy signal router 1, refutation persistence repos 1 (→ None, None), expert-review gate 1 (→ None), MLflow tracker recorder created; REDIS_URL dead port, MLflow file store in scratch, OPIK_ENABLED=false, discovery off |

## Read
The loaded synthetic backing is reachable ONLY through the planted-truth module seam, and through it the whole Lane A path (registry allowlist → provenance predicate → numeric coercion → one-hot with `__missing__` → curated DAG → LinearDML → refutation) recovers the planted effect at the CI width on the live rows. The deployed instance, which runs with the showcase flag on, serves none of the rows in real mode (in-process 503 with the flag set, and the container's own read of the table is empty). This is the same number the CI E2E measured on the generator frame (0.406, error 0.042 in PR #2228's D2b) — the live table reproduces the in-memory frame.

## Recorded, not asserted
- The 95% CI [0.3734, 0.4407] EXCLUDES the truth 0.364 — the LinearDML +0.04 bias PR #2228 documented (linear final stage on a step-function CATE; g-computation lands within 0.011). A follow-up for the estimator, not a Lane C defect; the CI is not a pass criterion.
- The response carries the "structural prior not applied (no feature_manifest_source)" warning twice — duplicated wording, pre-existing on this path.
- Full width (`--full-width`, k≈73) not run here (CI-shape only, as the E2E runs in CI).

## Files
`run.log` (stdout/stderr), `raw_live_ci_width.json` (every number above), `container_before.txt`, `container_after.txt`, `preload_dry_run.txt`, `owner_load_command.md`, `gates.md`.
