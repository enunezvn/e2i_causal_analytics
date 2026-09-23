# Post-Lane-B live probe — `optum_biologic_persistence` on the deployed API after #2230 + #2232 (2026-09-23)

**VERIFIED.** On the production container serving main `814eaa959` (PR #2230 Lane B and PR #2232 real-backed label merged; `e2i_api` started `2026-09-23T00:15:46Z`), one default-path `POST /causal/agent-analyze` on the real cohort completed, its response is labelled `data_source="database"` (was `"synthetic"` on the same request before #2232), and its estimate is identical to the pre-Lane-B cert run to full float precision — Lane B is dark on this dataset and Lane E's structural decider abstains on the machine-provenance attestations, exactly as the spec's §7 decision and the Lane B PR body state.

## The pair (same request: `treatment_dupixent → discontinued_180d`, `limit 20000`, discovery off by default)

| | pre-B (Lane A cert, container `c0860bbf4`, 2026-09-22 22:02Z) | post-B (this probe, container `814eaa959`, 2026-09-23 00:2xZ) |
|---|---|---|
| status | completed | completed |
| `data_source` | `synthetic` (showcase flag; PR #2232 cause) | **`database`** |
| estimator | LinearDML | LinearDML |
| ATE | 0.007056123869614513 | 0.007056123869614513 |
| 95 % CI | [−0.0053728049252601395, 0.019485052664489168] | identical |
| p | 0.26583454458398803 | identical |
| `dag_source` | domain_knowledge | domain_knowledge |
| `discovered_confounders` | [] | [] |
| container flipped during the run | no | no (`container_before` == `container_after` in `_probe`) |

Raw: `raw_post_b_discontinued_180d.json` (response + `_probe` block: request, poll timeline, container before/after); log: `probe.log`; script: `post_b_probe.py`. Pre-B raw: `docs/demos/results/2026-09-22_optum_biologic_persistence_cert/raw_discontinued_180d.json`.

## What the pair proves

1. **#2232 live**: the real-backed dataset's response label no longer follows the deployment flag (`E2I_INCLUDE_SYNTHETIC=true` is still set on the container).
2. **Lane B on this dataset**: no behaviour change, and the response says why, verbatim: `"structural prior not applied: the dataset declares no feature_manifest_source, so no authored structure can be matched to it"`. The estimate is bit-identical.
3. **Lane E under Lane B's provenance rule**: the response model carries none of `feature_role_panel`, `anchored_confounders`, `approved_structure_roles`, `approved_leak_exclusions`, `structural_prior` (they live in agent state, not in the serialised response — same limitation the Lane A cert recorded for `skipped_tests`), so the abstain-on-machine-attestation behaviour is proven in-process by the Lane B verifier (`verify_b_panel_compare.log`, PR #2230 body) and not observable from the API response. The discovery-on path that would have exercised Lane D/E/vote live is blocked by issue #2233.

## Findings (not fixed here)

- **F1** The `structural prior not applied` warning is emitted **twice** in one response (`warnings` has the identical string at two positions). The Lane B verifier's LOW noted the warning fires on every run; the duplication is new here. Cosmetic; worth a one-line dedupe.
- **F2** Response-model gap: the Lane B / Lane E channel fields are not serialised, so live verification of the structural path needs either an in-process run or a response-model extension (owner call).

## Deploy ledger (committed with this evidence)

`docs/demos/results/2026-09-15_trx_canonical/rollback/attempts/{causal_program_20260922,lane_c_20260922,lane_b_2232_20260922}.{pre,env,sha256}` — the three program deploys (44d310f0f run 35771791513 attempt 2; c0860bbf4 run 35781866332; 814eaa959 run 35796394678). The `.log` job logs are not committed (445 KB each; re-derivable from GitHub by `RUN_ID`/`JOB_ID` in the `.env`). A peer session recorded the 814eaa959 run under its own label (`lane_b_814eaa959`, same `RUN_ID`) 21 s earlier; left untouched.
