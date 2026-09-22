# Lane C — remibrutinib pre-wiring: disproofs and evidence (2026-09-22)

Spec: `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md` §3 "Lane C" (on Lane A's branch, not main). Base: origin/main `efaad6eeb`. Every number below cites the captured file:line in this directory.

## D1 — does main's matcher miss remibrutinib? (item 1's premise)

Assumption item 1 rests on: the claim-level converter does not recognise a remibrutinib (Rhapsido) fill today. Cheapest disproof: run main's `_csu_biologic_mask` / `_classify_biologic_brand` on one Rhapsido row, with a Xolair row as the positive control. Run against the unmodified main checkout (its converter is clean vs `efaad6eeb`, `d1_matcher_gap_on_main.txt:1-2`; Lane A's branch does not touch the file).

| measurement | value | cite |
|---|---|---|
| main's `CSU_BIOLOGIC_BRANDS` | `('XOLAIR', 'DUPIXENT')` | `d1_matcher_gap_on_main.txt:4` |
| main's `CSU_BIOLOGIC_GENERICS` | `('omalizumab', 'dupilumab')` | `d1_matcher_gap_on_main.txt:5` |
| `_csu_biologic_mask` on `RHAPSIDO / remibrutinib / 00078110030` | `[False]` | `d1_matcher_gap_on_main.txt:7` |
| `_classify_biologic_brand` on that row | `None` | `d1_matcher_gap_on_main.txt:8` |
| control: Xolair row | mask `[True]`, classify `xolair` | `d1_matcher_gap_on_main.txt:9` |

Premise survives: a remibrutinib fill is invisible to every cohort target (initiation / discontinuation / persistence) and to the brand classifier on main.

## D-NDC — does the placeholder NDC ever decide a classification on real data?

The spec names NDC `00078-1100-30`, which `src/ml/synthetic/clinical_codes.py` documents as a synthetic-demo placeholder on the Novartis labeler, not the marketed NDC. Risk: on a real drop a different Novartis product with product code 1100 would be matched. Cheapest disproof: how often do real medication rows carry a code without Brand_Name / Generic_Name (the only case where the NDC decides)?

| measurement (raw April 2026 drop, `data/rwd/csu/csu_data.xlsx`, sheet `medication`) | value | cite |
|---|---|---|
| medication rows | 10,000 | `d_ndc_names_null_rate_raw_drop.txt:3` |
| Brand_Name null / Generic_Name null / code null | 0 / 0 / 0 | `d_ndc_names_null_rate_raw_drop.txt:5` |
| code-only rows (both names null, code present) | 0 | `d_ndc_names_null_rate_raw_drop.txt:6` |
| rows whose code starts with the Novartis labeler `00078` | 0 | `d_ndc_names_null_rate_raw_drop.txt:7` |
| rows naming RHAPSIDO / remibrutinib | 0 / 0 | `d_ndc_names_null_rate_raw_drop.txt:8` |

Consequence for the design: the vocabulary matches the NDC at PRODUCT level only (`000781100` / `00078-1100`; the bare labeler is refused by a test), the names decide on every real row measured, and the marketed NDC is listed as an owner addition at the post-launch refresh (`src/data/csu_biologics.py` module docstring).

## D2 — is the full causal_impact graph CI-runnable on main for the planted cohort? (item 3's run shape)

Assumption item 3 rests on: the whole production path (registry -> loader -> submit -> the REAL graph) completes inside a CI per-test budget on the planted cohort (n = 3,000). Known from Lane A (its unmerged branch fixes it): on main the refutation node's DoWhy `identify_effect` burns its 100,000-iteration cap once the adjustment set has >= 17 members (467 s at k = 77). Probe: `probe_graph_runtime.py` (this directory), run in-process on this lane's tree with Supabase blanked, MLflow on a scratch file store and Redis on a dead port, one probe at a time under the box lock (`d2_probe_summary.txt:1-2`), at the planted-confounder width (`narrow`: age, charlson, payer one-hot -> k = 5) and at the full 64-feature contract width (`wide`: k = 73 after one-hot).

| run | wall | status | estimator | ATE (true) | 95% CI | refutation | cite |
|---|---|---|---|---|---|---|---|
| narrow n=2,000 k=5 | 83.5 s | completed | LinearDML (Auto) | 0.4136 (0.3619) | [0.372, 0.455] | 5/5 passed | `d2_probe_summary.txt:5`, `d2_probe_narrow_2000.json` |
| narrow n=3,000 k=5 | 85.0 s | completed | LinearDML (Auto) | 0.4072 (0.3643) | [0.373, 0.442] | 5/5 passed | `d2_probe_summary.txt:9`, `d2_probe_narrow_3000.json` |
| wide n=2,000 k=73 | 671.3 s | **failed** ("Refutation did not run — the effect is unvalidated") | LinearDML (Auto) | 0.4107 (0.3619) | [0.372, 0.449] | not run | `d2_probe_summary.txt:13`, `d2_probe_wide_2000.json` |

Split of the narrow run: estimator tournament (causal_forest, linear_dml, drlearner, ols) 14:37:18 -> 14:37:55 = 37 s, refutation suite 14:37:55 -> 14:38:33 = 38 s (`d2_probe_summary.txt:16-22`). Peak RSS 781-858 MB.

**Consequence for the run shape (the CI width decision).** The full width is NOT CI-runnable on main: 671 s, ending `failed` with no refutation — the identify-cap pathology Lane A fixes on its branch. The planted-truth test therefore runs in CI at the narrow shape (the three planted confounders plus five other baseline features, k ~ 13 after one-hot; `test_causal_csu_escalation_planted_truth.py` `CI_COVARIATES`) with `@pytest.mark.timeout(300)` — the unit lane's stall window is 600 s and `tests/unit/test_tests_meta/test_session_stall_watchdog_1655.py` requires window >= 2x the largest literal marker it collects, so 300 s is the ceiling; ~85 s measured here leaves ~3.5x headroom for a slower runner. The full 64-feature default runs through the SAME test under `E2I_CSU_PLANTED_FULL_WIDTH=1` (manual, or on every run once Lane A's identify fix merges).

**What the test asserts, and a finding it does not hide.** The spec's criterion is the POINT estimate within the generator's tolerance (0.10) of the planted RD-scale ATE; that is what is asserted. The 95% CI is NOT asserted to cover the truth, because measured it does not: all three runs sit +0.04..+0.05 above the truth with a CI that excludes it. Cheapest disproof of "the generator's truth is wrong" (`d2_estimator_vs_dgp_recovery.py` -> `.txt`): on the E2E test's own frame (n = 3,000, seed 20260922; truth 0.364, naive 0.498, `d2_estimator_vs_dgp_recovery.txt:1`) a probit g-computation with the DGP's structure recovers 0.375 (`:2`), IPW on the true propensity structure 0.387 (`:4`), AIPW 0.381 (`:5`), against the graph's LinearDML 0.407 (`:6`). A correctly specified adjustment lands within 0.011 of the truth, so the +0.04 is an estimator-side property (a linear final stage on a step-function CATE with treatment-variance weighting), not a generator defect. It is recorded here as a follow-up for the estimator-calibration work, not loosened away: the test still discriminates, the naive contrast misses by 0.13 (fails the tolerance) and the adjusted estimate by ~0.04 (passes).

## Gates

All in `gates_summary.txt` (line cites below); every pytest `-n 0 -p no:cacheprovider`, serialised under the box lock with a hard timeout; the E2E and every probe run with the unit tree's dead-Supabase pin plus an MLflow file store and a dead Redis port (write-free — the box is prod).

| gate | result | cite |
|---|---|---|
| `ruff check --no-cache src/ tests/` + touched scripts (CI's command) | All checks passed! | `gates_summary.txt:3` |
| `ruff format --check src/ tests/` + touched scripts (CI's command) | 3345 files already formatted | `gates_summary.txt:4` |
| pytest batch 1: 52 files = the 4 non-E2E lane files + every module importing a changed module | 1018 passed, 1 failed -> the failure was `test_causal_brands.py::test_load_frame_without_brand_does_not_filter` (a one-row frame hits the mirrored constant-treatment 400); fixed by taking Lane A's version of the test byte-for-byte | `gates_summary.txt:5-8` |
| pytest batch 2: E2E + registry + the two mirrored test files + `test_session_stall_watchdog_1655.py` (marker ceiling) + `test_module_size_ratchet.py` | 72 passed in 230 s (the E2E itself ~85 s per D2) | `gates_summary.txt:9-11` |
| teeth: `journey_brand` of remibrutinib set to `competitor` | `test_convert_optum_rwd_remibrutinib.py` RED (1 failed with `-x`) | `gates_summary.txt:13`, `gates_teeth_results.json` |
| teeth: one-arm `raise` removed from `select_csu_escalation_contrast` | `test_convert_optum_mart_csu_arms.py` RED: `test_one_arm_frame_is_refused` | `gates_summary.txt:14` |
| teeth: one column (`cci_hiv`) deleted from migration 149 | `test_csu_escalation_cohort_contract.py` RED: `test_migration_149_columns_equal_the_contract` | `gates_summary.txt:15` |
| teeth: the `<col>=__missing__` dummy disabled in the loader | `test_causal_csu_escalation_registry.py` RED: `test_synthetic_mode_loads_coerces_and_one_hots_the_backing` | `gates_summary.txt:16` |
| teeth: `confounders` / `modeled_confounders` emptied in the agent task state | E2E RED: the graph served the unadjusted 0.4976 (== the generator's naive_diff) and the tolerance assertion failed — the test fails exactly when the adjustment set is lost | `gates_summary.txt:17-19` |

| codex r1 MED: `propose_causal_questions` screened on RAW covariate names after the loader one-hot-expanded them (KeyError -> 500; pre-existing on main for the default dataset, newly reachable for csu_escalation_causal) | red-first test `test_causal_propose_categorical_covariates.py` RED before (KeyError on `geographic_region` / the seven Optum categoricals), GREEN after the fix; the candidate set equals the route's own loadable-pair enumeration (9/9 default, 4/4 csu) | `gates_codex_r1_fix.txt` |
| codex r2 MED: the per-drug `patient_journeys.brand` did not survive `_build_and_write_cohort` — `brand` is post-index (`OPTUM_FORBIDDEN_NON_TARGET`) and the leakage gate strips it from the model-frame parquet (since d72db4e7a; the old literal `competitor` never reached it either) | fixed WITHOUT touching the gate: a metadata sidecar `e2i_ml_v3_patient_journey_brands.parquet` (journey id, patient id, brand) written from the ungated journeys; red-first `TestWrittenArtifactBrand` reads both parquets back (RED: sidecar missing; GREEN: journeys brand-free, sidecar `Remibrutinib` / `competitor`), the converter family 328 passed | `gates_codex_r2_fix.txt` |

Every plant was restored by `cp` from a backup and sha256-verified (`gates_teeth_results.json` `restored_sha_ok`); `git status` after the run shows only the lane's own changes.

## Not done here (owner decisions, stated in the PR)

- Migration 149 rides the deploy on merge (prod schema write; rollback file provided).
- The synthetic backing rows are NOT loaded into prod `csu_escalation_causal` (needs the loader generalised from Lane A's `load_optum_causal_cohort.py` after #2220 merges, and an explicit GO). Until then real mode AND synthetic mode return 503 for the dataset; the E2E certifies the wiring from the generator's rows.
- The remibrutinib NDC prefix is the spec-named synthetic placeholder product code (00078-1100), product-level only; the marketed Rhapsido NDC is a post-launch addition (D-NDC shows the names decide on every real row measured).
- The CI run shape of the E2E is the narrow width (D2); the full width is behind `E2I_CSU_PLANTED_FULL_WIDTH=1` until Lane A's identify fix merges.
- The LinearDML +0.04 bias with a CI excluding the truth on this DGP (D2) is a recorded estimator-calibration follow-up.
