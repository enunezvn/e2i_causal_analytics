# HCP Brand-Adoption Gold-Standard Models — Design + Plan

**Date:** 2026-06-14
**Goal:** 3 HCP-grain models (`hcp_adoption_{brand}_goldstd_lr_v1` for Remibrutinib/Fabhalta/Kisqali)
populating `ml_performance_metrics` with real walk-forward + holdout trends — completing the
4 cohorts × 3 brands = 12-model gold-standard vision (9 patient-grain models already live).

**User-approved decisions (2026-06-14):**
- Population: **all 5,000 synthetic HCPs per brand** (15k (hcp,brand) rows), brand-specific propensity.
- Extras: **expose in Time-Series cohort selector** + **clean up superseded orphan models**.
- Hold all merge/deploy for explicit user / parallel-session direction.

## Architecture — maximal reuse, one genuinely new table

| Piece | Approach |
|---|---|
| DGP | Reuse `_compute_adoption(rng, centrality_z, brand)` (`src/ml/synthetic/generators/hcp_adoption_artifact.py:64`) — already per-brand (`_BRAND_ADOPT_SCALE`), leakage-safe (exogenous centrality → segment → confounded treatment_arm → adoption; leaky cols never emitted). **Add only a temporal `consideration_date` spread** over ~37 months. |
| Table `hcp_brand_adoption` (mig **076**) | grain = (hcp_id, brand). Cols: label `adopted` (0/1) + `adoption_category`, `consideration_date` (walk-forward axis), `data_split`, `is_synthetic`, FK→hcp_profiles. Features stay in `hcp_profiles` (joined at load = single SSOT). |
| Generator | New `hcp_brand_adoption_generator.py` → 5k HCP × 3 brands = 15k rows to prod. |
| Spec | `cohort_spec.make_hcp_spec(brand)`: `grain="hcp"`, `label_column="adopted"`, covariates = (peer_influence_score, influence_network_size, years_experience, specialty, geographic_region). |
| FeatureBuilder | HCP load path: query `hcp_brand_adoption ⋈ hcp_profiles`, **alias `consideration_date`→`journey_start_date`** so `walk_forward` (`_DATE_COL="journey_start_date"`, walk_forward.py:72), `recorder`, `cohort_deployer` are reused **untouched**. Encoder (`build_from_frame`/`transform`) is grain-agnostic, reused. |
| Harness | `run_hcp_cohorts.py` loops 3 brands, reuses `_run_one_cohort` (run_persistence_eval.py:116) verbatim. |
| Frontend | add "HCP Adoption" to the existing cohort dropdown (TimeSeries.tsx) → `hcp_adoption_{brand}_goldstd_lr_v1`. |
| Cleanup | deregister pooled `csu_initiation` / `pnh_persistence` / `pnh_discontinuation` + their metrics (FK-safe order). |

## Leakage-safety (the crux)
Each brand is its **own** model; within a brand each HCP appears **exactly once** → no train/holdout leakage.
DGP is stationary (adoption depends on centrality/segment, not month); `month` is never a feature; the leaky
cols (`days_to_first`, `first_adoption_dt`, `adopter_rank`) are never emitted. Walk-forward trains on HCPs
considered `< M`, evals on month `M` → genuine out-of-sample. Feasibility proven (throwaway sim AUC 0.77–0.82).

## Verified grounding (origin/main @ 5833df9c)
- `hcp_profiles` HAS: peer_influence_score, influence_network_size, years_experience, specialty,
  geographic_region, territory_id, sales_rep_id, adoption_category, is_synthetic — but **NO brand/temporal cols**.
- `CohortSpec` (cohort_spec.py): fields name, target, brand, label_column, grain, base_covariates;
  helpers `goldstd_model_name(cohort,brand)`/`goldstd_experiment_name(cohort,brand)` → `{cohort}_{brand_lower}_goldstd_{lr_v1|eval_v1}`.
- `_run_one_cohort(client, spec, *, model_name, experiment_name)` — train champion (train+validation) →
  holdout headline AUC → FK-safe clear → register staging → walk-forward → record both sources.
- `register_cohort_model(..., stage='staging')` keeps models out of production serving filter but resolvable
  by the Time-Series trend endpoint.
- Frontend selector builds `${cohort}_${brand.toLowerCase()}_goldstd_lr_v1` from cohort+brand dropdowns.
- Next free migration = 076 (core; highest = 075).

## Tasks (subagent-driven; see task tracker #27–34)
1. **T1** migration 076 hcp_brand_adoption (controller-authored schema contract).
2. **T2** temporal per-brand generator + tests.
3. **T3** make_hcp_spec + HCP FeatureBuilder load path + tests.
4. **T4** run_hcp_cohorts runner + tests.
5. **T5** frontend "HCP Adoption" cohort option + test.
6. **T6** orphan-model cleanup script + tests.
7. **T7** prod execution — apply mig, gen+load 15k, train+register 3 models live, verify, run cleanup.
8. **T8** scoped CI gates (cheapest-disproof) + PR (held).

## Acceptance
- 3 models registered (staging) + `ml_performance_metrics` populated (backtest_wf + holdout) per handle.
- Holdout AUC in ~0.74–0.83 band, real (no mocks), leakage-safe.
- Time-Series "HCP Adoption" × brand serves a real trend (live-verified via reviewer JWT).
- Orphan pooled models removed; registry holds only live per-brand (9 patient + 3 HCP) handles.
- CI green; PR held for merge/deploy.
