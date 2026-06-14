# P2 Persistence — feature-lock experiment (measured)

**Date:** 2026-06-14
**Question:** Which leakage-safe feature tier gives the best held-out AUC for the
all-brands synthetic **persistence** cohort (`persistent_180d`)? Lock
`FeatureBuilder.keep_columns` to the winner.

## Method (faithful)
- Real async Supabase client (`get_async_supabase_client`, env from the `e2i_api`
  container → host-reachable kong at `http://172.17.0.1:54321`).
- Real `FeatureBuilder` encoder (one-hot + median-impute + `__isna` flags).
- Load once (all candidate cols), split by `data_split`: fit on `train`+`validation`,
  score AUC on `holdout` (newest 3 months). Estimator = `LogisticRegression(
  class_weight='balanced', max_iter=1000)` (AUC is calibration-invariant, so the bare
  LR matches the calibrated run model's discrimination).
- All-brands (brand=None): persistence is brand-agnostic (pos rate Remibrutinib 0.542
  / Fabhalta 0.552 / Kisqali 0.545).

## Data
- 25,000 synthetic persistence-labeled rows. train+val n=8,336 (pos_rate 0.548);
  holdout n=15,211 (pos_rate 0.546). Well-balanced.

## Result (measured holdout AUC)
| Tier | encoded n_feat | holdout AUC |
|------|----------------|-------------|
| **A — base-3** (`disease_severity`, `academic_hcp`, `geographic_region`) | 9 | **0.5936** |
| B — A + `brand` | 13 | 0.5933 |
| C — A + `brand` + 16 leakage-safe baseline clinicals* | 48 | 0.5907 |

\* age_at_diagnosis, gender, age_group, insurance_type, risk_score, engagement_score,
urticaria_severity_uas7, prior_antihistamine_therapy, ecog_performance_status,
ldh_ratio, egfr, proteinuria_g_day, disease_stage, hr_status, her2_status,
complement_inhibitor_status.

## Decision
- **LOCK `keep_columns` = base-3** (= module default). Brand adds nothing (−0.0003);
  the 16 baseline clinicals add noise and slightly hurt (−0.0029). Mirrors P1
  initiation, where base-3 alone also beat all expansions.

## Caveat surfaced to the user (performance gate)
- ~0.59 holdout AUC is **weak** (initiation was 0.671 on the same covariates). It is
  the HONEST leakage-safe ceiling: baseline severity/HCP/region predict *whether a
  patient initiates* far better than *whether they persist 180 days* (persistence
  depends on tolerability/adherence/life-events not encoded pre-index). Richer
  leakage-safe features do not rescue it.
- Per the user's standing rule ("only generate new data if the models fail
  performance"), this measured ~0.59 is the trigger for a proceed-vs-regenerate
  decision. Discontinuation is the exact complement → same ~0.59.

## Update (2026-06-14): user chose REGENERATE (moderate boost) — DONE, re-measured 0.77

Root cause of the weak 0.59: in `cohort_outcomes.generate_discontinuation_outcomes`
the dominant outcome driver was `treatment_arm` (±1.2 logit, leakage-denylisted →
invisible to a leakage-safe model), while the observable confounders were tiny
(`severity` 0.18, `academic` −0.40) and `geographic_region` had **zero** effect
(pure noise), plus heavy `Normal(0, 0.5)` noise.

**DGP change (commit `e66d38e3`), causal treatment effect UNCHANGED:**
`_DISC_SEVERITY_COEF` 0.18→0.55, `_DISC_ACADEMIC_COEF` −0.40→−0.80, new
`_DISC_REGION_LOGIT` (midwest −0.9 / northeast −0.3 / south +0.3 / west +0.9),
noise 0.5→0.35, `_DISC_INTERCEPT` −0.85→−2.4 (re-tuned for the band). Locked by a
measured sim on the real covariates (conf/noise sweep → moderate target ~0.78).

**Targeted regeneration (NOT full re-gen — initiation/TRUE_ATE untouched):** applied
the new DGP to the existing 25k synthetic patients' covariates and UPDATEd only
`persistent_180d`/`discontinued_180d` (backup `~/db_backups/persist_backup_*.tsv`).
Verified on the live DB: disc 0.497 / persistent 0.503 (in-band), 0 complement
violations, region monotonic (midwest 0.320 → northeast 0.442 → south 0.568 →
west 0.663), `treatment_initiated` sum unchanged (8750).

**Re-measured holdout AUC on regenerated data:** base-3 **0.7747** (was 0.5936),
+brand 0.7748, +extras 0.7733 → base-3 still wins. `keep_columns` = base-3 stands.
Discontinuation is the exact complement → same ~0.77.
