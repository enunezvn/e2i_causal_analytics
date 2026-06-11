# Optum_enriched.parquet — readability, converter-compat, temporal defs, deployability disproof (2026-06-09)

**Scope:** review-only, no code change. Faithful no-mock measurements on the real file.
**File:** `data/rwd/Optum_Parquet/Optum_enriched.parquet` (40 MB, 814,587 rows × 205 cols)

## 1. Can we read it? — YES
- Patient-grain mart: **814,587 rows = 814,587 distinct `patid`** (one row per patient), all `entity_type=patient`.
- Therapeutic area = **CSU / anti-IgE biologics** (Xolair + Dupixent), same TA as the old mart.
- Read pattern: metadata-only schema reads + ≤74-col projections. Full-width (205-col) reads risk OOM on this 6.6 GiB box.

## 2. Old vs new file structure (the key compatibility fact)
| | Old `Optum.parquet` | New `Optum_enriched.parquet` |
|---|---|---|
| rows × cols | 3,758,007 × 252 | 814,587 × 205 |
| grains (`entity_type`) | patient 814,587 + **optum_hcp 2,753,238** + veeva_hcp 189,951 + market 231 | **patient only** (814,587) |
| HCP columns | unprefixed on the `optum_hcp` grain (`adoption_status`, `kol_score`, `shared_patient_*`, `referral_in_*`) | denormalized onto patient rows, **`primary_hcp_*` prefix**; network features absent |
| new patient columns | — | `total_*` lifetime aggregates, full Charlson+Elixhauser, data-quality scores |

The enriched file is the **patient grain of the same mart, feature-enriched** — it does **not** contain the stacked `optum_hcp` grain.

## 3. Converter compatibility
- **`convert_optum_mart.py`** (initiation / discontinuation / persistence): **works as-is** on the enriched file.
  - `entity_type` is all `patient` → the converter's `entity_type` gate is a no-op (passes everything). ✓
  - **MART_SAFE_FEATURES: 62/64 present**; the 2 "missing" (`enrollment_duration_days`, `geographic_region`) are **converter-derived** (not raw-read) and weren't in the old file either → full leakage-safe feature coverage. ✓
  - All gating + cohort-logic columns present (`index_date`, `treatment_start_date`, `last_observed_date`, `last_coverage_end`, `max_internal_gap_days`, `terminal_gap_days`, `index_biologic_brand`, `claim_record_count`, `elig_start_date`, `zipcode_5`). ✓
  - **Only change needed:** `DEFAULT_INPUT` (line 62) still points at the old file → pass `--input data/rwd/Optum_Parquet/Optum_enriched.parquet` or update the constant. No structural change.
- **`convert_optum_hcp_adoption.py`** (HCP adoption): **incompatible** with the enriched file.
  - Selects `entity_type == optum_hcp` (zero such rows in enriched) and reads **unprefixed** HCP columns; enriched has them `primary_hcp_*` and lacks the network features (`shared_patient_*`, `referral_in_*`) that drive the deployable HCP model.
  - **Keep using the old entity-stacked `Optum.parquet` for the HCP cohort** (its 2.75 M `optum_hcp` grain is the right, richer source). Do **not** repoint this converter at the enriched file.

## 4. Temporal definitions — sound
- **index_date: 100% populated** (`missing_index_date_flag=1` is 0.000%), range 2016-06-01 → 2024-12-31.
- **Lookback:** `elig_start_date` → `index_date` (`enrollment_duration_days`), `continuous_enrollment=1` for 100% (pre-filtered). The 64 baseline features are measured at the anchor (`index_date` for initiation; dx-index ≤ `treatment_start_date` for disc/persistence) → leakage-safe by positive enumeration.
- **Leakage caveat:** the enriched file's NEW columns — `total_*` lifetime aggregates, `pdc`, `covered_days` — are **post-index / over-treatment** and would leak if added as features. The current allow-list correctly excludes them; do not add them.

## 5. Cheapest disproof — leakage-safe 5-fold CV AUC ceiling (HistGBM, MART_SAFE_FEATURES only)
| Cohort | n (full) | positives | base rate | **CV AUC** | top single-feat AUC (leak check) | Verdict |
|---|---|---|---|---|---|---|
| **Initiation** (initiated biologic ≤180d of dx index) | 787,781 | 11,079 | 1.41% | **0.762 ± 0.006** | 0.631 (no leak) | **Deployable-range — strongest candidate** |
| **Discontinuation** (disc_180d, 90d-gap) | 15,209 | 1,673 | 11.0% | **0.616 ± 0.012** | 0.604 (no leak) | **Feature-bound (~0.62)** — confirms prior; borderline |
| **Persistence** (persistent_at_180d) | 15,209 | 7,217 | 47.5% | **0.547 ± 0.014** | 0.531 (no leak) | **Not deployable — near chance** |
| **HCP adoption** (adopted_target_brand) | from OLD file (2.75 M optum_hcp) | — | — | **~0.84–0.85** (existing decision doc) | referral-net 0.68–0.80 | **Deployable — but from OLD file, not enriched** |

Sampling note: initiation fit on all 11,079 positives + capped negatives (200 k) — AUC ceiling is robust to negative subsampling. Single-feature AUCs all < 0.80 → no leak detected in any cohort.

## 6. Bottom line
- **Initiation is the clear winner on the enriched file** (AUC 0.76, well-posed 1.4% incident outcome — the old degenerate 94.7%-positive problem is gone). Worth running through the full 5-gate tier-0 harness next.
- **Discontinuation** stays feature-bound (~0.62) even with the richer mart; **Persistence** is near-random on pre-index features (driven by unobserved post-index tolerability/response/access).
- **HCP adoption** remains deployable but must be built from the **old entity-stacked file**, not this enriched one.
- The AUC ceiling is the cheap *filter*: only Initiation (and HCP, from the old file) clear the discrimination floor and merit the (more expensive) full tier-0 gate run.

Repro: `PYTHONPATH=. .venv/bin/python docs/reports/optum_enriched_cohort_disproof_20260609/disproof.py <cohort>`
