# Persistence/Discontinuation DGP Enrichment — Design Spec

**Date:** 2026-06-21
**Status:** Approved (brainstorming), pending implementation plan
**Tracking:** Task T9 in `.claude/plans/2026-06-21-frontend-review-fixes.md`
**Branch:** `feat/t9-persistence-dgp-enrichment`

## Problem

Across the platform every gold-standard model sits at ~70% accuracy (AUC ~0.64–0.71).
A user reviewing `/feature-importance` asked why only **3 features** are ranked. Investigation
proved this is **not** a model or UI weakness — it is the **Bayes ceiling of the data-generating
process (DGP)**.

The persistence/discontinuation outcome is generated in
`src/ml/synthetic/generators/cohort_outcomes.py:91-101` as:

```
logit = -2.4
        + brand_cate_scale · seg_treat · treatment_arm      # causal effect
        + 0.55 · disease_severity
        - 0.80 · academic_hcp
        + region_pull (±0.9 across 4 regions)
        + Normal(0, 0.35)                                    # logit noise
p_disc       = sigmoid(logit)
discontinued = Bernoulli(p_disc);  persistent_180d = 1 - discontinued
```

Only **3 covariates** enter the equation. Therefore:
- A 3-feature model captures **100% of the recoverable systematic signal**; any additional feature
  is statistically independent of the label by construction (this is exactly why the 2026-06-14
  feature experiment found tiers B/C — more columns — scored *lower* on holdout: pure overfitting).
- The achievable AUC (~0.67–0.70) is pinned by the modest coefficient sizes (small spread in `p_disc`)
  plus the logit noise and the Bernoulli draw. Even an omniscient model cannot beat it.

So the 12 models are essentially **optimal for the DGP they were given**. To get higher accuracy with
richer features we must change the **recipe that generates the outcome** — make the outcome genuinely
depend on more real clinical drivers with a higher signal-to-noise ratio.

## Goal

Enrich the persistence/discontinuation DGP so:
1. The outcome depends on **7 leakage-safe covariates** (not 3), all clinically plausible persistence drivers.
2. The achievable AUC rises to a **realistic ~0.78–0.82**, varied across brands (e.g. ~0.78/0.80/0.82) —
   better than 0.70 but still credible to a pharma audience (real RWD persistence models ≈ 0.65–0.80).
3. `/feature-importance` consequently ranks **7** covariates, resolving the original complaint *for real*
   (the model truly uses them), not via a UI disclaimer.

**Non-goals (this round):** initiation and HCP-adoption outcome families. They are fast follow-ups that
reuse this exact recipe on their own equations (`patient_generator.py` initiation; `hcp_brand_adoption`).

## Decisions (from brainstorming, user-approved)

| Decision | Choice |
|---|---|
| Accuracy target | Realistic **~0.78–0.82**, varied per brand |
| Scope this round | **Persistence + discontinuation only** (6 models: 3 brands × {persistence, discontinuation}) |
| Driver set | **7 covariates** = existing {disease_severity, academic_hcp, geographic_region} + 4 new |
| New drivers | `age_at_diagnosis`, `insurance_type`, `comorbidity_burden`, `prior_therapy_lines` |
| New columns? | Reuse `insurance_type` + `age_at_diagnosis` (already generated at `patient_generator.py:220,225`); **one additive migration** for `comorbidity_burden` + `prior_therapy_lines` (the schema's `comorbidities[]` is unpopulated and `previous_treatment` is on `treatment_events`, not `patient_journeys`) |
| Causal safety | New drivers are **prognostic-only, independent of `treatment_arm`** |
| Calibration | **Measure, don't assume** — a harness tunes coefficients to the achieved AUC + prevalence |

## Design

### 1. Structural-equation change (`cohort_outcomes.py`)
Extend the discontinuation logit **additively**, leaving the causal term and existing confounders
byte-for-byte intact:

```
logit = intercept*                                          # RE-TUNED for prevalence band
        + brand_cate_scale · seg_treat · treatment_arm      # UNCHANGED — causal effect
        + 0.55 · disease_severity - 0.80 · academic_hcp + region_pull   # UNCHANGED confounders
        + β_age   · f(age_at_diagnosis)                     # NEW prognostic
        + β_ins   · insurance_pull(insurance_type)          # NEW prognostic
        + β_com   · comorbidity_burden                      # NEW prognostic
        + β_prior · prior_therapy_lines               # NEW prognostic
        + Normal(0, σ*)                                     # σ* ≤ 0.35 — a calibration lever (§3)
```

**Sign convention:** the equation's target is the **discontinuation** logit (as in the current code:
severity `+0.55`, academic `−0.80`). The §2 driver table states clinical direction on **persistence**;
the implementation applies the **matching negated pull on the discontinuation logit**. E.g. "commercial
insurance → more persistence" ⇒ a **negative** pull on the discontinuation logit. Get this sign right per
driver or the new terms will invert the intended clinical effect.

`intercept*` and `σ*` are re-tuned (see §3) so marginal prevalence stays in **[0.05, 0.60]** and the
achieved AUC lands in target.

### 2. The 4 new drivers — population + signal contract
Each driver must satisfy two contracts or it does nothing useful:

- **Prognostic-only / treatment-independent.** Each new driver is drawn **independently of `treatment_arm`
  assignment.** This guarantees the new terms raise *predictive* AUC without altering the *true ATE* or the
  confounding structure — causal estimation and HTE recovery are mathematically untouched. (The existing
  confounders severity/academic remain confounders; the 4 new ones are pure prognostic factors.)
- **Independent signal (anti-collinearity).** Each must carry variance not already explained by
  `disease_severity`; otherwise it adds no AUC. `comorbidity_burden` may be *partially* correlated with
  severity but retains an independent component.

Per-driver:

| Driver | Source column | Encoding | Realistic effect on persistence |
|---|---|---|---|
| `insurance_type` | `patient_journeys.insurance_type` (already generated) | one-hot | access gradient: commercial > Medicare > Medicaid |
| `comorbidity_burden` | NEW column (additive migration), Poisson-drawn 0–5 | numeric | higher burden → less persistence |
| `age_at_diagnosis` | `patient_journeys.age_at_diagnosis` (already generated) | numeric | monotonic persistence gradient |
| `prior_therapy_lines` | NEW column (additive migration), 0–3 | numeric | more prior lines → less persistence |

**Population note:** `insurance_type` and `age_at_diagnosis` are already drawn in `patient_generator.py` (lines
220, 225) but *after* the outcome call, so they don't yet feed the equation — the implementation hoists their
generation above the outcome call. `comorbidity_burden` and `prior_therapy_lines` are new columns, generated
fresh and drawn independently of `treatment_arm`.

### 3. Calibration harness — measure, don't assume
Hand-picking coefficients and hoping for 0.80 is exactly the premise-guessing the project forbids. Instead,
extend `src/ml/synthetic/dgp/recovery_probe.py` into a calibration/validation step that:
1. Generates a sample under candidate coefficients.
2. Fits the oracle/LR (same encoding as `FeatureBuilder`) and **measures** holdout AUC per brand + marginal prevalence.
3. Tunes the coefficient scales until each brand lands **AUC ∈ [0.78, 0.82]** (varied) and **prevalence ∈ [0.05, 0.60]**.
4. Emits the achieved numbers, which become **asserted test gates** (regression-locked, like the 2026-06-14 experiment).

### 4. Invariants — hard gates (all must pass before retrain/reseed)
- **True ATE recovery** within the existing tolerance (treatment term unchanged ⇒ ATE preserved).
- **Segment CATE heterogeneity** ordering intact (high > medium > low) — `/segment-analysis` depends on planted heterogeneity.
- **Prevalence** ∈ [0.05, 0.60]; **brand distinctness** retained (per-brand `brand_cate_scale` and varied AUC).
- **Leakage-safety**: the 4 new drivers are pre-index ⇒ added to `FeatureBuilder.KEEP_COLUMNS`, **never** `LEAKAGE_DENYLIST`.
- **Complement**: `persistent_180d == 1 - discontinued_180d` exactly.

### 5. Re-lock → retrain → reseed
1. Re-run the 2026-06-14-style feature experiment on the enriched data → confirm the 7 covariates beat
   subsets → **re-lock `KEEP_COLUMNS`** for the persistence/discontinuation cohorts (per-cohort; initiation/HCP unchanged this round).
2. Retrain the **6** persistence/discontinuation models via `run_persistence_eval` and record metrics.
3. Reseed the substrate so the live DB carries the enriched columns + new labels.
4. **Risk — `ml_drift_history` FK (RESTRICT) landmine:** `register_cohort_model` delete+reinsert is blocked
   by the drift-history FK after metrics exist (error 23503). Handle per the known surgical pattern (record
   vs existing model_id by name, do not blind-reregister) — see memory `goldstd_eval_confusion_roc_drifthistory_fk`.

### 6. Downstream validation (proof the loop closed)
- `/feature-importance` ranks **7** covariates for persistence cohorts (encoded count grows with the insurance one-hot).
- Predictive cohort-scoring still works (BentoML schema unaffected; depends on T5 restart being done).
- `/segment-analysis` HTE still populated; `/causal-analysis` estimation unchanged.
- Live AUC visibly ~0.80, varied per brand.

## Scope boundary & follow-ups
- **In scope now:** persistence + discontinuation (6 models), one structural equation (`cohort_outcomes.py`).
- **Follow-ups (same recipe):** initiation (×3, `patient_generator.py`), HCP adoption (×3, `hcp_brand_adoption`).

## Risks
1. **Calibration may not reach 0.80 with 7 prognostic drivers** without unrealistically large coefficients →
   the harness will reveal this; if so, negotiate target down or add one more driver. Measured, not assumed.
2. **Collinearity** of new drivers with severity → little AUC gain. Mitigated by the independent-signal contract + the harness.
3. **Reseed/retrain cost & the FK landmine** (see §5.4).
4. **Downstream breakage** if a new driver accidentally correlates with `treatment_arm` → ATE/HTE shift.
   Mitigated by the prognostic-only contract + the ATE/CATE invariant gates.

## Files (anticipated)
- `src/ml/synthetic/generators/cohort_outcomes.py` — extend equation, re-tune intercept/noise.
- `src/ml/synthetic/generators/patient_generator.py` (and/or callers) — populate the 4 driver columns with realistic, treatment-independent distributions.
- `src/ml/synthetic/dgp/recovery_probe.py` — calibration/validation harness + asserted gates.
- `src/mlops/gold_standard_eval/feature_builder.py` — `KEEP_COLUMNS` re-lock for persistence/discontinuation (per-cohort).
- `src/mlops/gold_standard_eval/run_persistence_eval.py` — retrain the 6 models.
- `database/migrations/0NN_persistence_drivers.sql` — additive `comorbidity_burden` + `prior_therapy_lines` columns on `patient_journeys`.
- Tests: DGP unit tests (equation, prevalence band, complement, treatment-independence), calibration gates, ATE/CATE invariant tests.
