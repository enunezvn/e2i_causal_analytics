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
