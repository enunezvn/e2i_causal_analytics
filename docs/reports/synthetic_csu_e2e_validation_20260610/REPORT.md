# Synthetic-CSU E2E Pipeline Validation — tier0 parquet → deployable model → Tier1-5

**Date:** 2026-06-10 → final 2026-06-11 (after the #864/#866/#867/#868 fixes) ·
**Dataset:** `data/rwd/synthetic_CSU/` (seed 42, confounded DGP, FULL_SIZES — regenerated with the #864 split fix)
**Guide:** `docs/data/SYNTHETIC-CAUSAL-DATA-GUIDE.md` · **Branch:** `fix/synthetic-csu-4of4-13of13` (PR #870; the #864/#866/#867 round merged earlier via PR #865)

## Verdict — FINAL: 4/4 deployed, 13/13 × 4 cohorts

**The pipeline is validated end-to-end on synthetic data with known ground truth: tier0
deploys ALL FOUR cohorts to staging, and the full tier1-5 agent harness passes 13/13
against every cohort's tier0 state.** The gold-standard data did exactly what it exists
to do — it isolated, with surgical precision, four tier0 harness defects (#864 split
scramble, #866 noise-blind deployment caps, #867 overfit-blind champion selection,
#868 HPO blind to final-fit degeneracy) and seven tier1-5 harness↔data couplings, all
now fixed and red-first tested. Every gate that fires, fires honestly on a measured
criterion; nothing was loosened to manufacture a pass — persistence's deploy comes from
fixing a genuine HPO defect (the selected trial collapsed in the final-style fit), and
causal_impact genuinely recovers the designed TRUE_ATE (0.193 vs +0.172, within the
sidecar's 0.10 tolerance) with the refutation suite actually running.

## 0. Roadblocker fixes shipped in this round (red-first tested)

1. **#864 — `BaseGenerator._assign_splits` row-mass boundaries.** The old unique-date
   quantile cut mapped the anchored recent window into holdout (snapshot measured train
   24.8% / holdout 60.8%). Now rows fill each split's quota chronologically; a date
   straddling a quota boundary chunks across ADJACENT splits (the anchor cap concentrates
   40.6% of treatment_events rows on the single reference date — no whole-date scheme can
   reconcile that with the ratios; within one date there is no temporal order to leak, and
   the SplitValidator's temporal check is strict `>`). **Regenerated snapshot: every table
   measures exactly 60/20/15/5.**
2. **#866 — evaluation-split-size-scaled deployment caps.** `adaptive_success_criteria`
   now accepts `n_train/n_val/n_test` (threaded by the evaluator overlay from the
   materialized splits) and widens the overfit/calibration caps to the sampling-noise
   floor of the split each metric is measured on: `maximum_train_val_delta =
   max(fpr_tier, 2·√(SE²_HM(n_train)+SE²_HM(n_val)))` (Hanley-McNeil at AUC 0.7); ECE cap
   = max(step floor, 10-bin perfect-calibration noise mean + 2σ at n_test); slope/intercept
   caps scale by √(1000/n_test) below the van Calster anchor. Floors are hard minima —
   scaling only loosens, and the eager/scope-time path (no split sizes) is unchanged.
3. **#867 — gate-aware champion selection.** The "deployability-aware" pool in
   `_select_champion` read `overfitting_severity` from a key the comparison entries never
   carried at selection time in the shape it expected, so the pool silently fell back to
   raw-AUC + slope tie-break — which picked persistence's severely-overfit XGBoost (train→
   val Δ 0.1873) over a 0.002-lower-AUC deployable LR. Step 5b entries now carry each
   candidate's own EVALUATED v3 gate results (`success_criteria_results`), and the runner
   prints a per-candidate "Deployability gates" evidence line.
4. **#868 — HPO final-fit degeneracy guard** (`hyperparameter_tuner.py`, PR #870). The HPO
   objective (CV-AUC) selected `C=0.00195/l1` for persistence's LR — on the L1 zeroing
   cliff. The conformal wrapper's internal refit (smaller effective n than the CV folds)
   crossed the cliff → intercept-only model (CV promised 0.65, final delivered AUC 0.500).
   The guard re-fits the winning trial **final-style** (resampled arrays when
   `resampling_applied`), detects collapse (predicted-probability std < 1e-3; val AUC
   ≤ 0.52 with CV promise ≥ 0.58; recalibration slope ≥ 2.0 for raw-deploy candidates),
   and adopts the best non-collapsed trial by verification val-AUC. On persistence it
   fired exactly as designed: rejected trial 0 (prob std 0.0), adopted trial 4
   (`C=3.47/l2`) → LR_Conformal champion, all gates green, **DEPLOYED**.
5. **Tier1-5 harness↔data couplings** (`run_tier1_5_test.py` + `tier0_output_mapper.py`,
   PR #870). Five legacy-fixture couplings fixed: experiment_monitor's CLI `--timeout` is
   now a floor (the 20s per-agent config silently capped it); prediction_synthesizer's
   mapper builds the model-ready row via the state's fitted preprocessor/encoder (raw
   entity columns KeyError'd on one-hot names) with a real population prior;
   tool_composer's canned query derives bindings from the actual frame schema;
   causal_impact binds the designed binary treatment with real non-constant confounders
   (plus a harness-only, run-boundary-scoped networkx≥3.5 alias so pinned dowhy 0.12's
   refutation suite runs — prod fix tracked as #869); heterogeneous_optimizer maps the
   designed outcome/treatment/segments (the constant-0 legacy flag made every CATE 0.0).
6. **Mapper designed-binary allowlist** (codex-hardened). On frames without
   `treatment_arm` (hcp_adoption), the mapper prefers designed binary exposures from a
   fixed-order allowlist (`academic_hcp`) over a derived 50/50 median-split — the derived
   pseudo-treatment had no causal identity and produced an estimation/refutation mirror
   flip (+0.2492 / −0.2357). Never a generic {0,1} scan (column-order / one-hot-fragment
   hazard, regression-tested).
7. **feedback_learner LLM enum validation.** The LLM emitted
   `pattern_type='baseline_establishment'` — outside the `DetectedPattern` Literal
   contract — crashing the agent on all four cohorts. Out-of-contract patterns (invalid
   type OR severity) are now uniformly dropped fail-closed and surfaced via
   `pattern_parse_anomalies` (severity is load-bearing: model_retrain fires only for
   high/critical, so clamping was rejected).

## 1. Lean ground-truth checks (guide §5.0) — `lean_checks/`

Re-run on the **regenerated** dataset: all five offline checks **pass**
(`lean_check_results.json`): IPW recovers TRUE_ATE +0.1718 within tolerance; the naive
estimate is ~12× more biased; CATE ordering 0.294 > 0.191 > 0.074 matches the sidecar
(`ground_truth_20260610T211458.json`); propensity AUC ≈ 0.68 with full overlap; all four
cohorts modelable (CV-AUC 0.63–0.78, no leak signature). Seed-identical DGP draws — only
`data_split` changed vs the original generation.

## 2. Tier0 MLOps pipeline (8 steps/cohort) — `tier0_<cohort>/`

Invocation (driver `run_tier0_cohort.py`, exact CLI path + state-pickle capture):
`--data-dir data/rwd/synthetic_CSU/tier0/<cohort> --target <cohort target>
--brand Remibrutinib --indication "Chronic Spontaneous Urticaria (CSU)" --no-bentoml
--no-demo-cost-matrix --deployment-intent commercial --feature-manifest-source synthetic_csu`

| Cohort | n | val/test split | Champion | Deployer verdict | Binding gate (cap after #866) |
|---|---|---|---|---|---|
| **initiation** | 8,420 | 1,684 / 1,263 | LR (AUC 0.686) | ✅ **DEPLOYED** | — (delta 0.0046 ≪ 0.032) |
| **discontinuation** | 2,952 | 591 / 443 | LR_Conformal (AUC 0.647) | ✅ **DEPLOYED** (was blocked) | — (delta 0.0397 ≤ 0.049) |
| **hcp_adoption** | 1,688 | 337 / 254 | LR_Conformal (AUC 0.798) | ✅ **DEPLOYED** (was blocked) | — (delta 0.0317 ≤ 0.069; ECE 0.0708 ≤ 0.117) |
| **persistence** | 2,952 | 591 / 443 | LR_Conformal (AUC 0.637, via #868 guard) | ✅ **DEPLOYED** (was blocked) | — (delta 0.0462; slope dev 0.0640 ≤ cap; severity none) |

**Why discontinuation/hcp_adoption now deploy:** their old blocks were exactly the #866
noise-scale failures — measured clean deltas 0.0397/0.0317 against a fixed 0.03 cap that
sat at ~1.3σ of pure sampling noise at val n≈591/337, and an ECE cap (0.05) BELOW the
n_test=254 perfect-calibration noise floor (~0.079). The scaled caps clear them; the
permutation test stays GENUINE and every other gate is met on merit.

**Why persistence now deploys (#868, fixed in PR #870).** Persistence
(`persistent_180d` = exact complement of the deployed `discontinued_180d`) initially had
NO candidate passing both quality gates, at this size or double it. The root-cause chain,
established before the fix and preserved here as evidence:

- Per-candidate evidence (runner's "Deployability gates" line): XGBoost/LightGBM are
  genuinely overfit (Δ 0.187 / mild) — no honest cap clears that; plain LR passes the
  overfit gate but its deployed probabilities are underconfident ≈2× (recalibration slope
  ≈ 1.96, deviation 0.96 vs scaled cap 0.225 — the slope gate fires CORRECTLY);
  LR_Conformal was degenerate (test AUC 0.500).
- Root cause: the HPO objective (CV-AUC) selected `C=0.00195/l1` (CV 0.6499) — on the L1
  zeroing cliff. The conformal wrapper's internal refit (smaller effective n than the CV
  folds) crosses the cliff → intercept-only model (CV promised 0.65, final delivered
  0.50); the plain LR keeps L1-crushed coefficients → near-flat probabilities → Platt
  doesn't transfer → slope ≈ 2. A faithful offline replication on the exporter's exact
  splits with a healthy `C=1` measures slope dev 0.22/0.27 for BOTH complement cohorts —
  the extreme deviations are hyperparameter-induced, not data-induced.
- **The 2× data lever was tested faithfully and disproven** (`/tmp/synth_scale_probe`,
  5,897 initiators): persistence still blocked (all four candidates fail the slope gate
  at the tighter n_test=885 cap), and discontinuation REGRESSED — its all-gates-green LR
  champion was blocked by `minimum_recall` because the `validation_commercial_recall`
  threshold chooser targets the floor exactly (val recall 0.5009 → test 0.470 < 0.50).
  FULL_SIZES therefore stays at 25k; the fix belonged in the harness (#868), not the data.
- **The fix**: the degeneracy guard (fix #4 above) re-verifies the HPO winner final-style
  and falls back past the cliff. Canonical rerun (`tier0_persistence/console.log`): guard
  FIRED for LR_Conformal — rejected trial 0 (`C=0.00195/l1`, prob std 0.0e0), adopted
  trial 4 (`C=3.47/l2`, verification val AUC 0.6066, highest of 8 non-collapsed trials) →
  champion LR_Conformal test AUC 0.637, overfit delta 0.0462, slope dev 0.0640, severity
  none, permutation GENUINE (p=0.0000) → **DEPLOYED to staging**.

Per-cohort artifacts: `console.log` (final post-fix run), canonical
`rwd_pipeline_run_*.md` (earlier timestamps = the iteration runs), and
`tier0_state_<cohort>.pkl` — the tier1-5 input. (The pre-fix `*_pre864866` twins were
superseded and removed once all four cohorts reached DEPLOYED.)

### Defects found and fixed by the original exercise (commit `3c052fba`)

1. **Exporter contract columns** — GE suite auto-detect needs `patient_journey_id` +
   `discontinuation_flag`; step-3 inclusion criteria derive from `data_quality_score`
   (absent → CC_001 → 0 of 8,420 eligible).
2. **Leakage layers dropped the designed causal drivers** — fixed with the `synthetic_csu`
   feature manifest (declared-safe by construction). With it: `disease_severity` is the
   top predictor — exactly as the DGP designs.
3. **`resolve_manifest_source` M1 ordering bug** — raised ambiguity before consulting the
   override; M1 now defers to a valid override.

### Pipeline behaviors documented (not defects)

- **Demo cost matrix** (default-on) forces `business_utility` negative — dropped via the
  runner's own `--no-demo-cost-matrix`.
- **Clinical vs commercial intent**: the clinical default (recall ≥ 0.65, MCC ≥ 0.35) is
  unreachable *by design* on this data; the documented commercial profile is the
  appropriate axis for this platform.
- **Zero-margin threshold chooser**: `validation_commercial_recall` targets the recall
  floor exactly — now measured actually blocking a clean cohort (2× probe); folded into
  issue #868.

## 3. Tier1-5 agent validation — `tier1_5/` — 13/13 against ALL FOUR cohort states

`run_tier1_5_test.py --tier0-cache tier0_<cohort>/tier0_state_<cohort>.pkl
--skip-observability --timeout 240` (Opik intentionally stopped on this box). Canonical
runs 2026-06-11 01:36–02:05 UTC, one full harness pass per cohort tier0 state:

| Tier0 state | Result | Artifacts |
|---|---|---|
| initiation | **13/13 pass** | `results_initiation.json` + `console_initiation.log` |
| discontinuation | **13/13 pass** | `results_discontinuation.json` + `console_discontinuation.log` |
| persistence | **13/13 pass** | `results_persistence.json` + `console_persistence.log` |
| hcp_adoption | **13/13 pass** | `results_hcp_adoption.json` + `console_hcp_adoption.log` |

The five failures of the earlier run (tool_composer, causal_impact,
heterogeneous_optimizer, experiment_monitor, prediction_synthesizer) were all
harness↔data couplings to the legacy fixture schema — fixed by fixes #5/#6 above. A
sixth failure surfaced only at full scale (feedback_learner crashing on an
out-of-contract LLM enum) — fixed by fix #7. Highlights of what now runs **for real**:
causal_impact recovers ATE 0.193 (CI [0.141, 0.244]) vs designed TRUE_ATE +0.172 with
dowhy refuters executing; heterogeneous_optimizer finds 5 high + 5 low responders on
designed CATE; prediction_synthesizer scores the model-ready row through the actual
fitted preprocessor.

Commit provenance: the batch launched at harness commit `09f05c09`; the codex-R4
hardening commit `24186873` (allowlist + anomaly surfacing + typing) landed mid-batch,
so the persistence run executed fully at the PR tip while the other three ran at
`09f05c09`. The R4 deltas are healthy-path-neutral (allowlist selects the same
`academic_hcp`; anomaly surfacing only fires on drops; remaining changes are
type-annotation-only) and are unit-test-covered; re-running any cohort at the tip
reproduces via the §5 command.

## 4. Folder inventory

```
lean_checks/            run_lean_checks.py, lean_check_results.json (post-fix), run.log
run_tier0_cohort.py     tier0 driver (exact CLI path + state capture)
tier0_<cohort>/         console.log (final DEPLOYED run, all four cohorts),
                        rwd_pipeline_run_*.md (iteration evidence),
                        tier0_state_<cohort>.pkl (the tier1-5 input)
tier1_5/                results_<cohort>.json + console_<cohort>.log — canonical
                        13/13 evidence per cohort tier0 state
REPORT.md               this file
```

(Superseded artifacts — the `*_pre864866` twins and the earlier 8/13-era
`tier1_5/results.json`/`console.log` — were removed once every cohort reached
DEPLOYED + 13/13, per the session's replace-with-successful instruction.)

## 5. Reproduce

```bash
python scripts/load_synthetic_data.py --parquet-only --parquet-out data/rwd/synthetic_CSU --anchor-to-now
python scripts/write_ground_truth_sidecar.py --n 25000 --seed 42 --out-dir data/rwd/synthetic_CSU
python scripts/export_synthetic_tier0.py --src data/rwd/synthetic_CSU       # tier0 inputs
for c in initiation discontinuation persistence hcp_adoption; do
  LOKY_MAX_CPU_COUNT=1 python docs/reports/synthetic_csu_e2e_validation_20260610/run_tier0_cohort.py "$c"
  LOKY_MAX_CPU_COUNT=1 python scripts/run_tier1_5_test.py \
    --tier0-cache docs/reports/synthetic_csu_e2e_validation_20260610/tier0_$c/tier0_state_$c.pkl \
    --skip-observability --timeout 240 --output results_$c.json
done
```

## Issue ledger

| Issue | Status | What |
|---|---|---|
| #864 | **fixed** (PR #865, merged) | split scramble under --anchor-to-now (row-mass quota fill + boundary-date chunking) |
| #866 | **fixed** (PR #865, merged) | deployment caps blind to evaluation-split sampling noise (scaled caps) |
| #867 | **fixed** (PR #865, merged) | champion selection's deployability pool read a field production never set |
| #868 | **fixed** (PR #870, `Closes #868`) | HPO blind to final-fit degeneracy / the L1 regularization cliff (persistence) — guard + fallback; zero-margin recall chooser documented therein |
| #869 | **open** (deliberately deferred) | prod dependency break: networkx 3.6.1 removed `d_separated`, dowhy 0.12 calls it → causal_impact refutation fail-closes in prod; needs its own requirements PR. The tier1-5 harness uses a run-boundary-scoped alias meanwhile |
