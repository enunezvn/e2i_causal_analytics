# Lane C live verify — `csu_escalation_causal` on the deployed API (2026-09-22)

**VERIFIED.** On the production container serving main `c0860bbf4` (PR #2228 merged; `e2i_api` started `2026-09-22T21:22:09Z`), with the deployment flag `E2I_INCLUDE_SYNTHETIC=true` set exactly as the container carries it, the new dataset answers **503** to estimation and its read predicate is forced to real rows only. No synthetic row can be served as real on this instance, and no environment variable can flip the planted-truth seam.

## What was deployed

| item | value |
|---|---|
| merge commit | `c0860bbf4` (Merge pull request #2228, `--merge`) |
| deploy run | 35781866332, attempt 1, `success` (ledger: `deploy_attempt.env`) |
| migration 149 | `149_csu_escalation_causal.sql` applied `2026-09-22 21:16:42+00`; `to_regclass('public.csu_escalation_causal')` resolves |
| rows in `csu_escalation_causal` | 0 (0 synthetic) — the synthetic backing is NOT loaded (PR #2228 owner decision 2: load not before this guard is on main; the load itself is still a separate owner GO) |
| container env | `E2I_INCLUDE_SYNTHETIC=true`, `ENVIRONMENT=production` |

## Live API probe (`live_probe.py` → `live_probe.json`, read-only, admin token)

| dataset | `GET /causal/brands` | `GET /causal/variables` | `POST /causal/agent-analyze` |
|---|---|---|---|
| `csu_escalation_causal` | 200, `brands: []` | 200, registry (treatment `treatment_remibrutinib`, 4 outcomes, 64-feature covariate allowlist) | **503** `DependencyError` envelope (error_id `6a498156`) |
| `optum_biologic_persistence` (control, table empty at probe time) | 200, `brands: []` | 200, registry (`treatment_dupixent`, same outcomes) | **503** `DependencyError` envelope (error_id `3cabab05`) |

The API's error envelope hides the loader's reason, and the container log carries only the span status (503), so the predicate proof below is in-process.

## In-process guard probe against the production database (`inprocess_guard_probe.txt`, read-only)

Run from the host with the real `.env` and `E2I_INCLUDE_SYNTHETIC=true` in the process environment, i.e. the container's condition:

```
E2I_INCLUDE_SYNTHETIC env: true | deployment_includes_synthetic(): True
datasets.PLANTED_TRUTH_RUN: False
serves_synthetic_rows(csu_escalation_causal) = False
serves_synthetic_rows(optum_biologic_persistence) = True
predicate csu   : [('eq', 'is_synthetic', False)]
predicate optum : []
csu_escalation_causal -> HTTPException 503 No usable estimation rows for the requested variables (treatment_remibrutinib -> discontinued_180d) in dataset 'csu_escalation_causal'.
optum_biologic_persistence -> HTTPException 503 No usable estimation rows for the requested variables (treatment_dupixent -> discontinued_180d) in dataset 'optum_biologic_persistence'.
```

Read: the deployment-wide toggle is on (`deployment_includes_synthetic()` is True), yet the synthetic-backed dataset's read predicate is `is_synthetic = false` regardless, the planted-truth seam is `False`, and the loader fails closed with the named 503. This is the verifier's MED-B closed on the deployed instance, not only in the unit environment. Both "modes" the handoff asked about are covered: real mode is the deployed default (503, predicate applied); synthetic (planted-truth) mode is unreachable from any environment (`PLANTED_TRUTH_RUN` is a module seam only a test's monkeypatch can set — verified by the Lane C verifier with four env spellings on 9b800e91f).

## Finding surfaced by this probe (not a Lane C defect; Lane A follow-up)

`serves_synthetic_rows("optum_biologic_persistence")` is **True** on this instance because the Optum dataset follows the deployment-wide flag, and `POST /causal/agent-analyze` labels its response `data_source = "synthetic" if serves_synthetic_rows(dataset) else "database"` (`src/api/routes/causal/agent.py:270`). Once the real 15,209-row cohort is loaded (every row `is_synthetic=false`, enforced by the loader), Lane A's live cert responses would carry `data_source="synthetic"` for real RWD — a plausible-wrong label. The Optum read also carries no provenance predicate under the flag (`predicate optum: []`), so the label is not merely cosmetic: it is the deployment rule applied to a dataset that the program defines as real-only. Recommended fix (opened as a separate PR): a real-backed counterpart to `_CAUSAL_SYNTHETIC_BACKED` whose reads always apply `is_synthetic = false` and whose label is always `database`, regardless of `E2I_INCLUDE_SYNTHETIC`.

## Not done here

- No rows loaded into `csu_escalation_causal` (owner GO still required; the guard is now on main, which was the precondition).
- The planted-truth E2E was not run against prod (by design: it reads only `is_synthetic=true` rows through the module seam in a test process).
