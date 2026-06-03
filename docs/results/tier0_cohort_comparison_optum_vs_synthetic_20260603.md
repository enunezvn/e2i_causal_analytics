# Tier0 Cohort Comparison — Gap-Enriched Optum vs. a Reliable (Modelable) Cohort

**Date**: 2026-06-03
**Purpose**: explain *why* the gap-enriched Optum initiation cohort cannot train a usable tier0 model while the reference cohort can — by comparing **data-set characteristics**, not just model performance. The leakage over-drop bug was fixed (PR #648); this analysis isolates what remains, which is a **data** limitation, not a pipeline one.

**Cohorts compared**
- **Optum gap-enriched / initiation** — real-world Optum claims, `data/rwd/optum_gap_enriched/initiation`, target `treatment_initiated` (CSU biologic initiation).
- **Reliable cohort** — the synthetic `clean` regime (`run_tier0_test.py --regime clean --feature-manifest-source synthetic`; `ml_patients` generator, positive_rate=0.70, signal_strength=1.4, noise_sd=0.03, signalized extras), the Phase-2 modelable baseline used to validate FIX 2.

> Numbers below are measured: Optum from the parquet + the gate-bypass run; the reliable cohort regenerated with the same `clean` params and from its trained-model run.

---

## Per-cohort profiles

### Cohort 1 — Optum gap-enriched / initiation (real-world)

**Identity & construction**
- **Source**: Optum administrative claims (real-world). Files: `data/rwd/optum_gap_enriched/initiation/e2i_ml_v3_patient_journeys.parquet` (+ `treatment_events`, `hcp_profiles`).
- **Indication**: Chronic Spontaneous Urticaria (CSU). **Brand frame**: competitor.
- **Index event**: CSU treatment-eligibility index date.
- **Target**: `treatment_initiated` — a CSU biologic fill (Xolair/Dupixent) in the prediction window `[index, index+180]` (`initiated_biologic_180d`).
- **Feature window**: strictly pre-index `[index−180, index−1]` with a 30-day biologic washout before index (disjoint from the outcome window).
- **Enrichment**: gap "HCP targeting / KOL-influence" features joined via the pre-index prescriber NPI — but only **8 of 1,294 patients (1%)** matched, so those columns are ~99% empty.
- **Documentation**: `data_dictionary.csv` formally specs ~23 features (with windows); the parquet carries ~140, so most are undocumented extras.

**Measured characteristics**
- 1,294 patients; **37 positive events** (2.86% prevalence; class ratio 1:34).
- ~136 raw features (≈5 numeric, ≈131 categorical/flag/count) + 3 nested text columns dropped.
- **EPV 0.27** (train-split 0.13) — far below the EPV≥2 floor.
- Feature density: **median non-zero fraction 0.000**; **85% of features <5% populated**.
- Signal: median single-feature AUC **0.510** (≈ noise); the only AUC=1.0 is the leaky target sibling; restoring all leakage-safe features lifts CV-AUC only 0.539→0.556.

**Pipeline verdict**: QC overall 0.93 (completeness 0.70 ❌), `data_sufficiency` **HARD_FAIL**, gate-bypassed model predicts all-negatives (test AUC 0.57, severe overfit). → **NOT MODELABLE** (data-volume + sparsity limit, not a pipeline defect).

### Cohort 2 — Reliable cohort (synthetic `clean` regime)

**Identity & construction**
- **Source**: synthetic — `SampleDataGenerator.ml_patients()` (legacy generator) via `run_tier0_test.py --regime clean`.
- **Generator params**: `positive_rate=0.70`, `signal_strength=1.4`, `noise_sd=0.03`, `signalize_extra_features=True`, `n=4000`, `seed=42`. The Phase-2 modelable baseline (locked val-AUC band).
- **Index/windows**: pre-index by construction; one leakage family injected by design and correctly dropped.
- **Target**: `discontinuation_flag` (generator outcome).
- **Manifest**: `synthetic` — declares pre-index legitimate predictors `days_on_therapy`, `hcp_visits`, `prior_treatments`, `borderline_genuine_feature` as Layer-1 declared-safe (the features the leakage immunity protects).

**Measured characteristics**
- 4,000 patients; **~1,336 positive events** (≈33% as regenerated / ≈44% as the pipeline realized it; balanced either way, class ratio ~1:2).
- **8 raw features** (≈3 numeric, ≈5 categorical/flag) — deliberately low-dimensional.
- **EPV 167** — ~80× above the floor.
- Feature density: **median non-zero fraction 0.800**; **0% of features <5% populated** (dense).
- Signal: median single-feature AUC **0.618**; **5 of 8 features >0.55**; top predictor `days_on_therapy` AUC 0.725.

**Pipeline verdict**: QC overall 0.99 (completeness 0.95 ✅), `data_sufficiency` **PASS**, LogisticRegression trains to **val AUC 0.912**, overfit Δ 0.016 (none), MCC 0.69, permutation p=0.000 (genuine signal). → **MODELABLE / "ready for production"**.

---

## Side-by-side

### A. Volume & class balance — *the dominant driver*

| Characteristic | **Optum gap-enriched** | **Reliable (synthetic clean)** | Ratio |
|---|---|---|---|
| Rows (patients) | 1,294 | 4,000 | 3.1× |
| **Positive events** | **37** | **~1,336** | **~36×** |
| Prevalence | **2.86%** (rare event) | ~33–45% (balanced) | — |
| Class ratio (pos:neg) | **1 : 34** | ~1 : 2 | — |
| Train-split events (60–75% slice) | ~16 | ~800–1,000 | — |

### B. Dimensionality & events-per-variable (EPV)

| Characteristic | **Optum gap-enriched** | **Reliable (synthetic clean)** | Ratio |
|---|---|---|---|
| Raw features (ex-target) | **~136** (+3 nested text) | **8** | 17× more (Optum) |
| Numeric / continuous | ~5 | ~3 | — |
| Categorical / flag / count | ~131 | ~5 | — |
| **EPV** (events ÷ raw features) | **0.27** (train-split **0.13**) | **167** | **~600×** |
| Sufficiency absolute floor (= 2·nf/prev, EPV≥2 rule) | **12,125 rows** | ~32 rows | — |
| EPV verdict (Vergouwe 2007 floor = 2) | **0.13 ≪ 2 → HARD_FAIL** | 167 ≫ 2 → PASS | — |

### C. Feature density / sparsity

| Characteristic | **Optum gap-enriched** | **Reliable (synthetic clean)** |
|---|---|---|
| Median feature density (non-zero fraction) | **0.000** | **0.800** |
| Min feature density | 0.000 | 0.659 |
| Features < 5% populated | **85%** of features | **0%** |
| Features < 1% populated | **80%** of features | 0% |
| QC completeness dimension | **0.70–0.75** (< 0.90 ❌) | **0.947** (✅) |

### D. Signal strength (single-feature folded AUC vs. target)

| Characteristic | **Optum gap-enriched** | **Reliable (synthetic clean)** |
|---|---|---|
| Median single-feature AUC | **0.510** (≈ noise) | **0.618** (real signal) |
| Max *legitimate* single-feature AUC | ~0.51–0.55 (the AUC=1.0 is the **leaky** target sibling `initiated_biologic_180d`) | **0.725** (`days_on_therapy`) |
| Features with AUC > 0.55 | 14 (mostly sparse artifacts) | **5 of 8 (63%)** |
| Features with AUC > 0.60 | 11 (sparse) | 4 |
| Top legitimate predictors | weak demographics only (clinical signal was post-index/leaky) | days_on_therapy 0.73, prior_treatments 0.68, hcp_visits 0.62 |
| Multivariable signal (restore all features → CV-AUC) | **0.539 → 0.556 (chance)** | n/a — model already strong |

### E. Temporal / experimental design (both clean — *not* the problem)

| Characteristic | **Optum gap-enriched** | **Reliable (synthetic clean)** |
|---|---|---|
| Feature lookback window | strict pre-index `[index−180, index−1]` (+ 30-day washout) | generator-time pre-index by construction |
| Outcome window | `[index, index+180]` (disjoint from features) | disjoint by construction |
| Genuine post-index leaks present | yes — `*_180d` family (correctly dropped) | one injected leak family (correctly dropped) |
| Leakage design verdict | **clean** (windows disjoint) | clean |

### F. Pipeline outcomes & model performance

| Stage / metric | **Optum gap-enriched** | **Reliable (synthetic clean)** |
|---|---|---|
| QC overall_score | 0.93 ✅ | 0.99 ✅ |
| **data_sufficiency gate** | **HARD_FAIL** (EPV 0.13, n ≪ 12,125) | **PASS** |
| Model selected | LogisticRegression | LogisticRegression |
| Validation ROC-AUC | **0.57** *(gate-bypassed; near chance)* | **0.912** |
| Train→val/test AUC Δ (overfit) | **0.17–0.46 — severe** | **0.016 — none** |
| Predicts positives? | **No — "predicts ALL negatives"** | Yes (recall 0.82) |
| Precision / Recall / F1 | ~0.03 / ~0.40 / — | 0.83 / 0.82 / 0.83 |
| MCC | ≈ 0 | 0.69 |
| PR-AUC | ≈ prevalence (no lift) | 0.896 |
| Brier (calibration) | — | 0.118 |
| Permutation p-value (genuine signal) | — | 0.000 |
| **Deploy verdict** | **NOT MODELABLE** | **Ready for production** |

---

## Why the difference — it's compound, not one factor

A usable model needs **enough events to estimate each parameter** *and* **features that carry signal**. Optum fails on **both axes simultaneously**, and the two failures multiply:

1. **Events (36× fewer).** 37 positives vs ~1,336. With only ~16 events in the training split, almost nothing can be estimated reliably.
2. **Dimensionality (17× more features).** ~136 raw features vs 8. More parameters to fit on far fewer events.
3. **EPV (~600× worse).** 0.13 vs 167. The standard floor is EPV ≥ 2 (Vergouwe 2007 "severe-problems" minimum); Optum is **~15× below the floor**, the reliable cohort **~80× above** it. This is why the sufficiency gate's "≈12k-row floor" is correct, not arbitrary — it's the EPV=2 rule re-expressed: `2 × 136 / 0.029 ≈ 12k`.
4. **Sparsity.** 85% of Optum's features are <5% populated (median non-zero fraction **0.000**) — they are mostly absent, so they add dimensionality (cost) without information (benefit). The reliable cohort's features are dense (median 0.80).
5. **No multivariable signal.** Restoring *every* leakage-safe Optum feature lifts CV-AUC only 0.539 → 0.556 — chance. The clinically predictive variables were post-index (correctly dropped as leaks); the leakage-safe survivors are low-power demographics. The reliable cohort has 5/8 features individually above AUC 0.55 and a real top predictor at 0.73.

**Net:** Optum tries to fit ~136 parameters on ~16 events worth of near-zero-signal, mostly-empty columns → the model memorizes noise (train AUC ~0.98) and fails to generalize (test AUC 0.57), defaulting to all-negative predictions. The reliable cohort fits 8 dense, signal-bearing features on ~1,000 events → AUC 0.912 with no overfit.

## What this is *not*

- **Not a leakage-handling defect.** PR #648 fixed the over-drop; declared-safe pre-index features are now retained (Optum n_features 114→125). It changed nothing about modelability because there was no recoverable signal to retain.
- **Not a temporal-design defect.** Both cohorts use strict pre-index windows disjoint from the outcome; the only Optum drops are genuine post-index outcomes (`*_180d`).
- **Not an over-strict gate.** The `data_sufficiency` HARD_FAIL is correct — verified by bypassing it and observing a useless, severely-overfit model.

## What would make Optum modelable

The binding constraint is **treated-patient volume**, not features or thresholds:

1. **More events** — a materially larger extract. At 2.9% prevalence the EPV-2 floor is ~12k rows *for the current feature width*; far fewer rows suffice if dimensionality is also reduced.
2. **Fewer, denser features** — drop the 80–85% of columns that are <5% populated; keep a small set of well-populated, knowable-at-index predictors. Reducing to ~8–15 effective features drops the EPV-2 floor by ~10×.
3. **A higher-base-rate target or cohort** — a less rare outcome (or a cohort enriched for it) raises events per row directly.

Until one of these holds, the tier0 gates will (correctly) decline to ship a model on this cohort.

---

## Root-cause forensics — is the unmodelability a data bug? (2026-06-03)

**Question asked:** Is this cohort unmodelable because the parquet was *not properly converted, joined, or processed* (a pipeline bug), or because the raw extract genuinely lacks signal (a data limitation)?

**Method:** A 13-agent forensic team traced six feature families end-to-end (each independently re-deriving ground truth, reading the converter code, and re-running the joins on a sample of cohort patients), with every verdict adversarially challenged by a skeptic agent trying to *overturn* it. The decisive claims were then independently re-verified by hand against the raw parquet.

**Verdict: GENUINE DATA LIMITATION — not a conversion/join/processing bug (high confidence).** The converter, join keys, time windows, and dtypes are correct. The all-zero / all-null columns faithfully represent records that **do not exist in the raw `Optum_Parquet` drop**.

**The proof it's the data, not the code** — features built by the *same* machinery populate correctly, so a bug would have broken them too:

| Control (same loops / joins / window) | Result | Proves |
|---|---|---|
| `has_asthma` (J45), `has_depression` (F32/33), `has_angioedema` (T78.3) | 4 / 3 / 1 patients | comorbidity diag-matching works |
| `hospitalizations_total` (inpatient, pre-index window) | 72 / 1294 nonzero | inpatient join + window work |
| `charlson_score` | 38 / 1294 nonzero | dx scoring works |
| lab window counts (`free_t4`=20, `tsh`=18) | reproduce to the integer | lab join + window are exact |

**Why each empty family is the *data*, not the pipeline:**

| Family | Root cause in the raw extract | Bug? |
|---|---|---|
| 28 drug-fill cols (h1/h2/ltra/steroids/immunosupp) | `medication.parquet` is **biologic-dispensing-only** — `Generic_Name ∈ {omalizumab:1598, dupilumab:705}`; zero non-target drugs exist, and the biologics present are the *target* (excluded as leakage) | Genuine |
| 11 office/ED/provider cols | `procedure.parquet` is **Xolair-administration-only** (single code `j2357`, **zero E&M codes**) and **has no `npi` column** → office-visit counts can only be 0 and the provider-specialty join is structurally impossible; `inpatientdata` is **facility-only** (no `ED` in `tos_cd`) | Genuine |
| 4 comorbidity flags (atopic derm, allergic rhinitis, autoimmune thyroid, NSAID-hypersensitivity) | dx comes **only** from inpatient `diag1-5` + L50-filtered demographics — **no outpatient diagnosis table** exists; the specific codes (L20/J30/E06.3) are outpatient-typical and absent from the pre-index inpatient window | Genuine |
| Lab features | `lab.parquet` is **index-forward** — 0 of 856K rows are pre-index; ANA's mapped LOINC has 0 rows | Genuine (+ the code-label defect below) |
| Gap enrichment (8 `treating_hcp_*`, 8/1294) | genuine sparse HCP linkage for a treatment-naïve cohort (64 → 20 pre-index fill → 12 have `npi='na'` sentinel → 8 real); the separate vendor "Gap features" drop is **deliberately not joined** (different index anchor, post-index outcomes = leakage; tested ≤ chance CV-AUC) | Genuine |
| Provenance | **not** a `--max-patients`/pilot truncation (cohort patids span the full demographics range; 5000 → 1294 is documented clinical attrition); a ~9-year real span | Genuine |

**One confirmed defect (real, but immaterial to modelability):** the converter's `CSU_LABS_LOINC` map (`scripts/convert_optum_rwd.py`) mislabeled three analytes — `6206-7` (peanut IgE) was labeled *eosinophil*, `3051-0`/`3053-6` (free/total T3) were labeled *TPO antibody*, and `26453-1` (RBC) was labeled *CBC*. So three lab features measured the wrong analyte. This is a genuine correctness bug worth fixing for future/broader extracts, but it **does not change the unmodelable conclusion** — even correctly coded, none of these labs enrich for the 37-event target (all at/below the 2.86% base rate). *(Fixed 2026-06-03 with a behavioral regression test.)*

**Bottom line:** this **confirms and sharpens** the conclusion above. The cohort is unmodelable not merely because of 37 events, but because the raw extract is **concept-scoped** (meds/procedures are target-biologic-only), **single-source for diagnoses** (inpatient-only), **index-forward** (no pre-index labs), and **event-poor**. No re-conversion of *this* parquet can populate the missing covariate families. The genuine remediation is **upstream data acquisition**: request a re-extract with general pre-index pharmacy, an outpatient/professional claims table (with NPIs + dx codes), an ED/outpatient-facility table, and pre-index labs.

---

## Appendix — full per-feature breakdown

Every feature in each cohort, sorted by single-feature folded AUC (descending). Columns:
- **Type**: `num` (continuous), `num-disc` (numeric, ≤12 distinct), `categ` (≤25 categories), `hi-card` (high-cardinality / ID — not scored).
- **Card.**: distinct non-null values (0 = column is entirely null).
- **Non-null %**: fraction of rows with a value.
- **Density**: fraction of rows with a **non-zero** value (numeric) or non-null (categorical) — the "is this feature actually present" measure.
- **Single-feat AUC**: folded `max(AUC, 1−AUC)` of that one feature against the target; `—` when unscorable (constant, all-null, or high-cardinality ID).

**How to read Appendix A (Optum):** only the post-index **leak** (`initiated_biologic_180d`, AUC 1.000 — correctly dropped) and one **2%-populated** lab (`free_t4_result_last`, 0.763 on a tiny subset) clear AUC 0.63; the best **dense, legitimate** features (`age_at_index`, `dx_l50_9_count`, insurance/payer/plan demographics) top out at ~0.59–0.63. Below row ~28 the table is two failure modes stacked: **sparse** clinical/lab/HCP features (0.1–3% density, AUC ≈ 0.50 = noise) and a large block of **entirely-constant/empty** columns (cardinality 0–1, 0% density — the full medication-history `*_fill_count`/`*_days_supply` and visit-count families are all-zero in this cohort). ~60 of 136 features carry literally no information; ~40 more are sparse noise. **Appendix B (synthetic)** is the inverse: 4 of 8 features are 100%-dense and individually predictive (0.62–0.73).



#### Appendix A — Optum gap-enriched / initiation (all features, sorted by single-feature AUC, descending)

| # | Feature | Type | Card. | Non-null % | Density | Single-feat AUC | Note |
|---|---|---|---|---|---|---|---|
| 1 | `initiated_biologic_180d` | num-disc | 2 | 100% | 2.9% | 1.000 | **post-index leak** |
| 2 | `free_t4_result_last` | num | 16 | 2% | 1.5% | 0.763 |  |
| 3 | `insurance_product` | categ | 2 | 100% | 100.0% | 0.629 |  |
| 4 | `payer_category` | categ | 2 | 100% | 100.0% | 0.629 |  |
| 5 | `payer_bus_raw` | categ | 2 | 100% | 100.0% | 0.629 |  |
| 6 | `dx_l50_1_count` | num-disc | 2 | 100% | 17.0% | 0.621 |  |
| 7 | `age_at_index` | num | 71 | 100% | 100.0% | 0.621 |  |
| 8 | `primary_diagnosis_code` | categ | 3 | 100% | 100.0% | 0.616 |  |
| 9 | `plan_type` | categ | 6 | 100% | 100.0% | 0.613 |  |
| 10 | `payer_product_raw` | categ | 6 | 100% | 100.0% | 0.613 |  |
| 11 | `age_group` | categ | 4 | 100% | 100.0% | 0.605 |  |
| 12 | `treating_hcp_match_count` | num-disc | 2 | 100% | 0.6% | 0.594 |  |
| 13 | `dx_l50_9_count` | num-disc | 2 | 100% | 66.4% | 0.591 |  |
| 14 | `data_quality_score` | num | 13 | 100% | 100.0% | 0.557 |  |
| 15 | `geographic_region` | categ | 4 | 93% | 93.4% | 0.534 |  |
| 16 | `dx_l50_8_count` | num-disc | 2 | 100% | 16.6% | 0.530 |  |
| 17 | `charlson_score` | num-disc | 6 | 100% | 2.9% | 0.515 |  |
| 18 | `hospitalizations_total` | num-disc | 6 | 100% | 5.6% | 0.515 |  |
| 19 | `elixhauser_score` | num | 23 | 100% | 3.6% | 0.512 |  |
| 20 | `data_split` | categ | 4 | 100% | 100.0% | 0.511 |  |
| 21 | `crp_tested` | num-disc | 2 | 100% | 0.9% | 0.510 |  |
| 22 | `urban_rural_code` | categ | 2 | 100% | 100.0% | 0.509 |  |
| 23 | `tsh_tested` | num-disc | 2 | 100% | 1.4% | 0.507 |  |
| 24 | `gender` | categ | 2 | 100% | 100.0% | 0.507 |  |
| 25 | `free_t4_tested` | num-disc | 2 | 100% | 1.5% | 0.506 |  |
| 26 | `tpo_ab_tested` | num-disc | 2 | 100% | 0.8% | 0.504 |  |
| 27 | `ige_total_tested` | num-disc | 2 | 100% | 0.5% | 0.503 |  |
| 28 | `mental_health_flag` | num-disc | 2 | 100% | 0.4% | 0.502 |  |
| 29 | `has_asthma` | num-disc | 2 | 100% | 0.3% | 0.502 |  |
| 30 | `asthma_claim_count` | num-disc | 3 | 100% | 0.3% | 0.502 |  |
| 31 | `atopy_score` | num-disc | 2 | 100% | 0.3% | 0.502 |  |
| 32 | `eosinophil_tested` | num-disc | 2 | 100% | 0.3% | 0.502 |  |
| 33 | `has_depression` | num-disc | 2 | 100% | 0.2% | 0.501 |  |
| 34 | `depression_claim_count` | num-disc | 3 | 100% | 0.2% | 0.501 |  |
| 35 | `cbc_tested` | num-disc | 2 | 100% | 0.2% | 0.501 |  |
| 36 | `has_anxiety` | num-disc | 2 | 100% | 0.2% | 0.501 |  |
| 37 | `anxiety_claim_count` | num-disc | 2 | 100% | 0.2% | 0.501 |  |
| 38 | `dx_angioedema_count` | num-disc | 2 | 100% | 0.1% | 0.500 |  |
| 39 | `has_angioedema` | num-disc | 2 | 100% | 0.1% | 0.500 |  |
| 40 | `angioedema_claim_count` | num-disc | 2 | 100% | 0.1% | 0.500 |  |
| 41 | `patient_journey_id` | hi-card | 1294 | 100% | 100.0% | — |  |
| 42 | `patient_hash` | hi-card | 1294 | 100% | 100.0% | — |  |
| 43 | `lookback_start_date` | hi-card | 983 | 100% | 100.0% | — |  |
| 44 | `primary_diagnosis_desc` | categ | 1 | 100% | 100.0% | — |  |
| 45 | `zip_code` | hi-card | 1122 | 100% | 100.0% | — |  |
| 46 | `data_source` | categ | 1 | 100% | 100.0% | — |  |
| 47 | `ingestion_timestamp` | categ | 1 | 100% | 100.0% | — |  |
| 48 | `updated_at` | categ | 1 | 100% | 100.0% | — |  |
| 49 | `zip5` | hi-card | 1122 | 100% | 100.0% | — |  |
| 50 | `zip3` | hi-card | 427 | 100% | 100.0% | — |  |
| 51 | `dx_total_csu` | num-disc | 1 | 100% | 100.0% | — |  |
| 52 | `months_since_first_dx` | num-disc | 1 | 100% | 100.0% | — |  |
| 53 | `csu_chronicity` | categ | 1 | 100% | 100.0% | — |  |
| 54 | `split_config_id` | categ | 1 | 100% | 100.0% | — |  |
| 55 | `payer_lis_dual_raw` | categ | 1 | 44% | 44.0% | — |  |
| 56 | `tsh_result_last` | num | 18 | 1% | 1.3% | — |  |
| 57 | `tpo_ab_result_last` | num-disc | 9 | 1% | 0.8% | — |  |
| 58 | `treating_hcp_targeting_decile_max` | num-disc | 2 | 1% | 0.6% | — |  |
| 59 | `treating_hcp_priority_tier_best` | num-disc | 2 | 1% | 0.6% | — |  |
| 60 | `treating_hcp_kol_score_max` | num-disc | 6 | 1% | 0.6% | — |  |
| 61 | `treating_hcp_kol_score_100pt_max` | num-disc | 7 | 1% | 0.6% | — |  |
| 62 | `treating_hcp_influence_network_size_max` | num-disc | 7 | 1% | 0.6% | — |  |
| 63 | `treating_hcp_kol_category_top` | categ | 2 | 1% | 0.6% | — |  |
| 64 | `ige_total_result_last` | num-disc | 7 | 1% | 0.5% | — |  |
| 65 | `crp_result_last` | num-disc | 7 | 1% | 0.5% | — |  |
| 66 | `treating_hcp_is_specialist_any` | num-disc | 2 | 1% | 0.3% | — |  |
| 67 | `tsh_abnormal_flag` | num-disc | 2 | 1% | 0.2% | — |  |
| 68 | `cbc_result_last` | num-disc | 3 | 0% | 0.2% | — |  |
| 69 | `ige_total_abnormal_flag` | num-disc | 2 | 1% | 0.1% | — |  |
| 70 | `eosinophil_result_last` | num-disc | 2 | 0% | 0.1% | — |  |
| 71 | `eosinophil_abnormal_flag` | num-disc | 2 | 0% | 0.1% | — |  |
| 72 | `crp_abnormal_flag` | num-disc | 2 | 1% | 0.1% | — |  |
| 73 | `tpo_ab_abnormal_flag` | num-disc | 2 | 1% | 0.1% | — |  |
| 74 | `cbc_abnormal_flag` | num-disc | 2 | 0% | 0.1% | — |  |
| 75 | `state` | categ | 0 | 0% | 0.0% | — |  |
| 76 | `risk_score` | categ | 0 | 0% | 0.0% | — |  |
| 77 | `source_match_confidence` | categ | 0 | 0% | 0.0% | — |  |
| 78 | `source_stacking_flag` | num-disc | 1 | 100% | 0.0% | — |  |
| 79 | `source_combination_method` | categ | 0 | 0% | 0.0% | — |  |
| 80 | `source_timestamp` | categ | 0 | 0% | 0.0% | — |  |
| 81 | `data_lag_hours` | categ | 0 | 0% | 0.0% | — |  |
| 82 | `discontinued_180d` | categ | 0 | 0% | 0.0% | — | **post-index leak** |
| 83 | `persistent_at_180d` | categ | 0 | 0% | 0.0% | — | **post-index leak** |
| 84 | `discontinuation_flag` | categ | 0 | 0% | 0.0% | — | **post-index leak** |
| 85 | `payer_health_exch_raw` | num-disc | 1 | 100% | 0.0% | — |  |
| 86 | `has_atopic_dermatitis` | num-disc | 1 | 100% | 0.0% | — |  |
| 87 | `atopic_dermatitis_claim_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 88 | `has_allergic_rhinitis` | num-disc | 1 | 100% | 0.0% | — |  |
| 89 | `allergic_rhinitis_claim_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 90 | `has_thyroid_autoimmune` | num-disc | 1 | 100% | 0.0% | — |  |
| 91 | `thyroid_autoimmune_claim_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 92 | `has_nsaid_hypersensitivity` | num-disc | 1 | 100% | 0.0% | — |  |
| 93 | `nsaid_hypersensitivity_claim_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 94 | `office_visits_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 95 | `office_visits_allergist` | num-disc | 1 | 100% | 0.0% | — |  |
| 96 | `office_visits_dermatology` | num-disc | 1 | 100% | 0.0% | — |  |
| 97 | `office_visits_pcp` | num-disc | 1 | 100% | 0.0% | — |  |
| 98 | `ed_visits_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 99 | `ed_visits_urticaria_angio` | num-disc | 1 | 100% | 0.0% | — |  |
| 100 | `unique_providers` | num-disc | 1 | 100% | 0.0% | — |  |
| 101 | `h1_1g_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 102 | `h1_1g_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 103 | `h1_1g_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 104 | `h1_1g_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 105 | `h1_2g_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 106 | `h1_2g_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 107 | `h1_2g_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 108 | `h1_2g_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 109 | `h2_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 110 | `h2_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 111 | `h2_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 112 | `h2_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 113 | `ltra_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 114 | `ltra_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 115 | `ltra_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 116 | `ltra_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 117 | `sys_steroid_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 118 | `sys_steroid_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 119 | `sys_steroid_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 120 | `sys_steroid_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 121 | `top_steroid_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 122 | `top_steroid_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 123 | `top_steroid_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 124 | `top_steroid_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 125 | `immunosupp_ever_filled` | num-disc | 1 | 100% | 0.0% | — |  |
| 126 | `immunosupp_fill_count` | num-disc | 1 | 100% | 0.0% | — |  |
| 127 | `immunosupp_days_supply_total` | num-disc | 1 | 100% | 0.0% | — |  |
| 128 | `immunosupp_days_since_last_fill` | categ | 0 | 0% | 0.0% | — |  |
| 129 | `free_t4_abnormal_flag` | num-disc | 1 | 2% | 0.0% | — |  |
| 130 | `ana_tested` | num-disc | 1 | 100% | 0.0% | — |  |
| 131 | `ana_result_last` | categ | 0 | 0% | 0.0% | — |  |
| 132 | `ana_abnormal_flag` | categ | 0 | 0% | 0.0% | — |  |
| 133 | `specialist_concentration` | categ | 0 | 0% | 0.0% | — |  |
| 134 | `primary_specialist_type` | categ | 0 | 0% | 0.0% | — |  |
| 135 | `saw_allergist_flag` | num-disc | 1 | 100% | 0.0% | — |  |
| 136 | `saw_dermatologist_flag` | num-disc | 1 | 100% | 0.0% | — |  |

#### Appendix B — Reliable cohort / synthetic clean (all features, sorted by single-feature AUC, descending)

| # | Feature | Type | Card. | Non-null % | Density | Single-feat AUC | Note |
|---|---|---|---|---|---|---|---|
| 1 | `days_on_therapy` | num | 335 | 100% | 100.0% | 0.725 |  |
| 2 | `prior_treatments` | num-disc | 5 | 100% | 80.0% | 0.677 |  |
| 3 | `hcp_visits` | num | 19 | 100% | 100.0% | 0.619 |  |
| 4 | `data_quality_score` | num | 501 | 100% | 100.0% | 0.618 |  |
| 5 | `age_group` | categ | 3 | 100% | 100.0% | 0.599 |  |
| 6 | `geographic_region` | categ | 4 | 100% | 100.0% | 0.543 |  |
| 7 | `brand` | categ | 3 | 100% | 100.0% | 0.505 |  |
| 8 | `patient_journey_id` | hi-card | 4000 | 100% | 100.0% | — |  |
