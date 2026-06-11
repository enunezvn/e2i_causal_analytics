# Tier-0 Optum-mart DISCONTINUATION — first end-to-end result (2026-06-06)

## Context
First tier-0 run of the new `discontinuation_mart` cohort (Option B, strict 90d-gap
target). Two `data_preparer` bugs had to be fixed first (own PR,
`fix/data-preparer-remediation-loop`) before the cohort was evaluable:
1. **Remediation infinite-loop** → `GraphRecursionError(25)`: a perpetually-skipped
   action never made progress and the loops compounded past the recursion limit.
   Fix: no-progress stop (0 effective actions → halt, manual_required) + size
   `recursion_limit` to the configured remediation depth (80).
2. **Param-less action skipped on malformed LLM params**: the LLM emitted
   `params: reason="..."` (unparseable) on a `drop_column` for `cci_severe_liver`
   (rare comorbidity, perfect-class-separation at 11% prevalence). `drop_column`
   does not READ params, so the Codex MEDIUM-C skip was over-conservative → the
   column was never dropped → QC never passed. Fix: exempt param-less actions
   (`drop_column`/`deduplicate`) from the malformed-params skip (`impute` still
   skips — it reads `strategy`).

## Run
`run_optum_tier0_test.py --cohort discontinuation_mart --feature-manifest-source
optum_mart --single-model --hpo-trials 3 --no-bentoml` on the FULL cohort
(15,209 patients / 1,673 positives / **11.0% prevalence**; train 9,125).

## Result — feature-bound, genuine-but-weak (deployer correctly fail-closed)
| Metric | Value |
|---|---|
| ROC-AUC (val / test) | 0.598 / 0.608 |
| PR-AUC | 0.162 (baseline ≈ 0.11 → ~1.5× lift) |
| Precision / Recall | 0.16 / 0.56 |
| Permutation test | **signal=GENUINE**, p=0.0000, shuffled AUC 0.501 |
| Stratified 5-fold | AUC 0.613 ± 0.008 |
| Overfit (train→val AUC Δ) | 0.032 |
| honest band | in_band (not leakage-inflated, not noise) |
| Top predictor | `payer_category` |
| QC gate | PASSED (cci_severe_liver dropped) |
| Deployer | **BLOCKED — success_criteria_not_met** (0.598 < 0.65 AUC bar) |

## Interpretation
- The pipeline is now **fully functional** on disc — it trains, evaluates, and
  **fail-closes honestly** instead of crashing.
- The signal is **genuine but weak** (permutation-confirmed real, 5-fold stable),
  ROC-AUC ~0.60 — below the 0.65 deployment bar.
- **Same conclusion as initiation: feature-bound.** Despite disc's far healthier
  class balance (11% vs initiation's 1.41%) and adequate EPV (~26), the 64 baseline
  demographic+comorbidity features cap performance at AUC ~0.60. More events / better
  balance did NOT lift the ceiling — the binding constraint is the feature SET, not
  sample size or prevalence (cf.
  `tier0_optum_mart_initiation_events_disproof_20260606.md`).
- The lever remains **richer pre-index features** (raw-claims trajectories, prior-Rx,
  HCP/market signal), not cohort/prevalence engineering.

## Artifacts
- Run log: `/tmp/disc_tier0_v3.log` (both fixes active). Earlier logs:
  `/tmp/disc_tier0.log` (recursion crash), `/tmp/disc_tier0_fixed.log` (loop fixed,
  QC fail-close pre param-less-fix).
- Cohort: `data/rwd/mart/discontinuation/` (15,209 / 1,673).
- Fixes: `src/agents/ml_foundation/data_preparer/nodes/qc_remediation.py`,
  `src/agents/ml_foundation/data_preparer/agent.py` (recursion_limit).

*Generated 2026-06-06. persistence_mart (47.5% prev) not yet run.*
