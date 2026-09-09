# Lane 1 — impact discovery run on the NEW image (after PR #1986 deployed)

- Image: `ghcr.io/enunezvn/e2i-api:989eec83d4334d0fd7447d7805c36272baac3286` (`impact_image.txt`; merge commit 989eec83d, deploy run 34394202788, cert `cert_task14.txt` 16/16)
- Job: `4eb0c464-3f8c-4b69-89c7-f5b41e49fdc0` (POST /api/causal/discover-effects, dataset `patient_journeys`, brand `Remibrutinib`, the SAME committed `run_discovery.py` as the baseline), started 2026-09-09T19:46:55Z (`impact_started_at.txt`), 11/11 after 1590s (`impact.log`)
- Live DB before (`impact_db_before.txt`): pending|39 rejected|1; causal_validations 1080; discovered_dags 12 — after (`impact_db_after.txt`, 2026-09-09T20:14:08Z): pending|39 rejected|1; causal_validations 1135 (+55 = 11 runs × 5 recorded tests); discovered_dags 23 (+11)
- Bands: {'proceed': 5, 'block': 6} — **0 REVIEW** again. Non-critical tests now SCORE on 11/11 rows: `data_subset` 11/11 scored (5 recorded effects each; 10 passed, 1 FAILED on treatment_arm → persistent_180d), `bootstrap` 11/11 scored (20 recorded effects each, all passed; `stopped_for_budget` false everywhere). The plan's "50 effects" expectation is the runner DEFAULT (`RefutationRunner.DEFAULT_CONFIG` bootstrap `num_bootstraps: 50`); the agent-analyze route overrides it to 20 (`src/api/routes/causal.py:3360`, `"bootstrap": {"num_bootstraps": 20}`) and the recorded `num_bootstraps` is 20 on every row — recorded here, not changed (the chat brain uses 10, `agent.py:310`).
- All six BLOCK rows re-used the same six pre-existing pending review rows as the baseline (review ids identical per pair); no new review rows were minted by the job (pending 39 → 39).
- The one non-critical FAILED (data_subset on treatment_arm → persistent_180d, the BLOCK-band pair whose brand-less twin is the lane's probe) changed no band: the row was already BLOCK on a sensitivity FAILED (critical). Its confidence with a critical failure is BLOCK regardless of the non-critical scores, so this row cannot demonstrate REVIEW; REVIEW needs a run with no critical failure, a sensitivity WARNING and BOTH non-critical tests failing — none of the 11 questions produced that.

| question | baseline band | impact band | data_subset | bootstrap | sensitivity | review decision |
|---|---|---|---|---|---|---|
| copay_support → adherent_180d | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |
| copay_support → low_gap_180d | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |
| copay_support → persistent_180d | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| psp_enrolled → adherent_180d | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| psp_enrolled → persistent_180d | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| rep_detailing_high → treatment_initiated | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| sample_dropped → treatment_initiated | block | block | skipped → passed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| treatment_arm → persistent_180d | block | block | skipped → failed (5 effects) | skipped → passed (20 effects) | failed | pending_review |
| treatment_arm → treatment_initiated | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | passed | None |
| trigger_accepted → treatment_initiated | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |
| urticaria_severity_uas7 → persistent_180d | proceed | proceed | skipped → passed (5 effects) | skipped → passed (20 effects) | warning | None |

REVIEW rows: []
