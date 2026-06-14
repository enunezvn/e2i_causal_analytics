# EXPERIMENT — Lock the initiation cohort feature set (Task 3)

**Date:** 2026-06-14
**Branch:** `feat/gse-p1-initiation`
**Decision recorded into:** `src/mlops/gold_standard_eval/feature_builder.py` (`KEEP_COLUMNS`)

## Assumption + falsifier

- **Assumption:** a leakage-safe feature set drawn from `patient_journeys`
  (+ patient-keyed `feature_values`) trains an initiation model with **holdout
  AUC materially > 0.5**.
- **Falsifier:** holdout AUC ≤ ~0.55, or any chosen feature is leaky.

The decision is the **measured holdout AUC**, not theory.

## Data (REAL rows, read-only, user-authorized)

Pulled from the prod docker DB (`supabase-db`) via `\copy ... TO STDOUT WITH CSV`
for `brand='Remibrutinib' AND is_synthetic=true`, split by `data_split`:

| split      | n    | `treatment_initiated` mean |
|------------|------|----------------------------|
| train      | 2103 | 0.3514 |
| validation | 750  | 0.3613 |
| test       | 492  | 0.3232 |
| holdout    | 5075 | 0.3513 |

One row per patient (8420 distinct patients = sum of split rows).

## Candidate columns + filters applied

**Schema reality checks that changed the plan:**

- `geographic_region` is the **coarse 4-region enum** (northeast/midwest/south/west),
  NOT the 40-territory field. The Task-2 cardinality-ceiling concern (>20 distinct →
  drop/coarsen) **does not apply** here; no coarsening needed. (The 40 territories
  live in `hcp_profiles.territory_id`, a different table.)
- Variance filter dropped near-constant encoded columns:
  `risk_score` (100% null), `age_group`/`gender`/`data_quality_score`/
  `journey_duration_days`/`payer_category` (100% null),
  `prior_antihistamine_therapy` (single value).
- `feature_values` is keyed by `entity_values->>'patient_id'`; only **3 of the 8**
  patient-keyed features are genuinely new (`comorbidity_count`,
  `prior_treatment_count`, `insurance_tier`). The other 5 duplicate
  `patient_journeys` columns (disease_severity, age_at_diagnosis, engagement_score)
  or are leaky (treatment_propensity, outcome_probability). The new 3 cover only
  ~66% of the brand's patients (left-join → ~34% nulls, median-imputed).

## Leakage scan (`src/data/feature_contract.py` `knowable_at` semantics)

`post_index`-knowable columns are FORBIDDEN as features. Confirmed in
`LEAKAGE_DENYLIST` (knowable only AFTER the initiation decision):
`days_to_treatment, discontinued_180d, persistent_180d, adherence_rate,
refill_count, gap_days, is_churned, treatment_arm` and the two outcome-derived
`feature_values` `outcome_probability`, `treatment_propensity` (post_index
causal-graph quantities, not pre-decision covariates). The label
`treatment_initiated` is dropped as `y`.

**Kept candidates (all pre-decision / index-or-enrollment knowable):**
`disease_severity`, `academic_hcp`, `geographic_region` (base),
`age_at_diagnosis`, `engagement_score`, `insurance_type`,
`urticaria_severity_uas7` (PJ extras),
`comorbidity_count`, `prior_treatment_count`, `insurance_tier` (feature_values).

## Train + eval method (memory-guarded)

`free -h` ≥ ~5 GiB available before each fit. `FeatureBuilder.build_from_frame`
on TRAIN → fit `sklearn LogisticRegression(class_weight='balanced',
max_iter=1000)`; HOLDOUT encoding **reindexed to the train feature_columns**
(one-hot columns differ across splits — this alignment is what makes the AUC
meaningful) and scored with `roc_auc_score`.

## MEASURED results (the decision)

Three tiers, fit on TRAIN, scored on HOLDOUT (n_train=2103 pos≈0.3514,
n_hold=5075 pos≈0.3513):

| tier | columns | holdout AUC |
|------|---------|-------------|
| **A** | base covariates only (`disease_severity, academic_hcp, geographic_region`) | **0.6709** |
| B | A + leakage-safe PJ extras (`age_at_diagnosis, engagement_score, insurance_type, urticaria_severity_uas7`) | 0.6694 |
| C | B + new patient-keyed feature_values (`comorbidity_count, prior_treatment_count, insurance_tier`) | 0.6659 |

**Cross-split stability of Tier A** (fit on TRAIN, scored on each held set):
validation **0.6850**, test **0.6431**, holdout **0.6709**. LR coefficients:
`disease_severity` 0.296, `academic_hcp` 0.417, region one-hots ≈ 0 (region
barely contributes but is harmless and is a codebase-intent covariate).

No perfect-separation artifact (AUC ≈ 0.67, not ≈ 1.0) → no hidden leakage.

## Decision

- Falsifier did **not** trigger: AUC 0.6709 > 0.6, > 0.5; no leaky feature kept.
- The 3 codebase-intent base covariates **alone** give the best held-out AUC;
  PJ extras and feature_values add noise, not signal (cheapest-disproof outcome
  — the simplest leakage-safe set is also the best).
- **LOCK Tier A:** `KEEP_COLUMNS = ("disease_severity", "academic_hcp",
  "geographic_region")` (== `INITIATION.base_covariates`).

`FeatureBuilder.build_from_frame` now restricts raw columns to `KEEP_COLUMNS`
(when set). Verified the locked set reproduces holdout AUC **0.6709** through the
real `build_from_frame` (fit) → `transform` (apply) API with an identical
train/eval column space.

## Train/eval alignment decision (CRITICAL design check)

`build_from_frame` recomputed `feature_columns` per call → train/eval column sets
silently disagreed (the experiment had to reindex by hand). Implemented a
minimal **fit/transform split** that Tasks 4/8 depend on:

- `build_from_frame(train)` = **fit**: learns numeric medians + the ordered
  `feature_columns`.
- `transform(eval)` = **apply**: encodes with the **train** medians and
  **reindexes to the fitted `feature_columns`** (absent one-hot cols → 0.0,
  unseen cols → dropped). Raises if called before fit.

Unit tests added in `tests/unit/test_mlops/test_gold_standard_eval/test_feature_builder.py`
(reindex alignment incl. unseen/absent categories, train-median imputation on
eval, transform-before-fit raises, KEEP_COLUMNS restriction). **6/6 pass.**

## Reproduction

Data pull (per split):
```
docker exec supabase-db psql -U postgres -d postgres -P pager=off -c \
 "\copy (SELECT patient_id, data_split, treatment_initiated, disease_severity,
   academic_hcp, geographic_region FROM patient_journeys
   WHERE brand='Remibrutinib' AND is_synthetic=true AND data_split='<split>')
   TO STDOUT WITH CSV HEADER" > pj_<split>.csv
```
Then fit `build_from_frame(train)` → LR → score `transform(holdout)` with
`roc_auc_score` (script output pasted in the Task-3 report).
