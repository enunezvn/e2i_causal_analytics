# Optum Mart Data Dictionary — `data/rwd/Optum_Parquet/Optum.parquet`

**Generated:** 2026-06-08 (from the parquet's embedded schema + lineage columns; read-only).
**Why this exists:** no dictionary file ships with the mart. This reconstructs one from the file's own metadata so the column set, grains, and — critically — the **comorbidity lookback window certification status** are documented in one place. Exhaustive machine-readable column+type list: **`docs/data/optum_mart_column_schema.csv`** (252 rows).

> This is the **INPUT mart** dictionary (252 cols, the vendor/upstream artifact). It is distinct from the per-cohort `data_dictionary.csv` our converters **emit** (the 64-feature tier-0 contract). See `docs/OPTUM_MART_CONVERSION.md` and `docs/reports/optum-mart-data-treatment-findings-20260608.md`.

---

## Provenance (from embedded metadata)

| Property | Value |
|---|---|
| Builder | **Apache Spark 3.5.5** (`org.apache.spark.version`) — an **internally engineered mart**, not a raw Optum vendor delivery |
| Pipeline version | `versions = ['9.1']` |
| Extract month(s) | `extract_ym_values` ∈ `{202509, 202512}` (Sept + Dec 2025 extracts) |
| Timezone | GMT |
| Rows × cols | 3,758,007 × 252 |
| Row groups | 4 |
| Type histogram | 87 integer · 54 long · 51 string · 25 double · 22 date · 7 array&lt;string&gt; · 6 timestamp |
| Git history | none — the parquet is untracked; **the build pipeline is not in this repo** |

**Implication:** there is no external "Optum data dictionary" to fetch. The authoritative spec for any upstream-engineered column (notably the comorbidity lookback window) is the **Spark mart-build pipeline v9.1**, owned by whoever produced this drop — request that, not Optum.

---

## Entity grains (`entity_type`)

The mart is **entity-stacked**: one file, four unjoinable grains. A column is populated only for the grain(s) it belongs to.

| `entity_type` | rows | grain | column families that apply |
|---|---|---|---|
| `optum_hcp` | 2,753,238 | one claims-derived HCP | identity · network · volume · provider-attrs · adoption |
| `patient` | 814,587 | one CSU patient | identity · clinical-dates · enrollment · quality · **comorbidity** · treatment-outcome · demographics |
| `veeva_hcp` | 189,951 | one CRM HCP | identity · provider-attrs · **engagement/trigger** |
| `market` | 231 | one brand×market cell | market TRx / share / business-metric |

---

## Column families

Full enumerated list with Spark types is in `optum_mart_column_schema.csv`. Families below; **bold** = leakage-relevant or load-bearing for the tier-0 converters.

### Identity & keys (all grains)
`patid` · `family_id` · `npi` · `hcp_id` · `hcp_npi` · `hcp_name` · `prov` · `dea` · `brand` · `molecule` · `therapeutic_area` · `entity_type`

### Clinical dates — patient (date32)
**`index_date`** (= `first_csu_dx_date`, 100% of patient rows) · `first_csu_dx_date` · `last_csu_dx_date` · **`elig_start_date`** · **`elig_end_date`** · **`treatment_start_date`** · **`last_observed_date`** · **`last_coverage_end`** · first/last `_xolair_{medication,proc,treatment}_date` · `first_rescue_steroid_date` · `first_urticaria_angioedema_ed_date` · `first_biologic_switch_date` · `first_immunosuppressant_addon_date`

### Enrollment / coverage / adherence — patient
`continuous_enrollment` · `covered_days` · `pdc` · `adherent_flag` · `discontinued_flag` · `maintained_flag` · `days_after_last_treatment_observed` · `payer_continuous_enrollment` · `non_continuous_enrollment_flag` · `has_csu_specialist_visit`

### Data-quality — patient
`diagnosis/procedure/cost/enrollment_quality_score` · `claim_record_count` · `diagnosis_record_count` · `procedure_record_count` · `cost_record_count` · `missing_*_flag` (diagcode/index_date/proc_date/proc_code/cost/elig_start/elig_end) · `has_{std,secondary,optional_pharmacy}_cost_flag` · `data_quality_score` · `data_quality_band` · `data_quality_issue_summary`

### **Comorbidity — patient** (the leakage-sensitive family)
`cci_*` (17 Charlson one-hots) · `elx_*` (31 Elixhauser one-hots) · `charlson_score` · `elixhauser_van_walraven_score` · `comorbidity_diag_distinct_count` · `comorbidity_diag_claim_count` · `charlson_risk_band` · `elixhauser_risk_band` · `high_comorbidity_burden_flag` · `comorbidity_source_tables` (array). **Window: see certification section below.**

### **Index-biologic & treatment outcome — patient** (converter targets derive here)
`index_biologic_brand` · `index_biologic_molecule` · `index_biologic_match_method` · `followup_90d_flag` · `max_consecutive_biologic_coverage_days` · `persistence_60d_flag` · **`max_internal_gap_days`** · **`terminal_gap_days`** · `discontinued_90d_flag` · rescue-steroid / ED / switch / immunosuppressant counts+flags · `refractory_flag` · `inadequate_flag` · `controlled_flag` · `csu_response_proxy_category` · `treatment_response` · `outcome_indicator`

### Demographics — patient
`gdr_cd` · `age_at_index` · `yrdob` · `zipcode_5` · `payer_category` · `payer_product` · `payer_bus` · `health_exchange_flag` · `lis_dual_flag`

### HCP network — optum_hcp
`treated_patient_count` · `influence_network_size` · `shared_patient_edge_count` · `shared_patient_weight` · `max_shared_patient_edge_weight` · `shared_patient_kol_score_pct` · `referral_{in,out}_degree` · `referral_{in,out}_patient_count` · `max_referral_in_edge_weight` · `referral_kol_score_pct` · `kol_score` · `kol_score_100pt` · `kol_category` · `influence_network_{source,method}`

### HCP provider attributes — optum_hcp / veeva_hcp
`bed_sz_range` · `cred_type` · `grp_practice` · `hosp_affil` · `prov_state` · `prov_type` · `provcat` · `taxonomy1` · `taxonomy2` · `specialty_group` · `specialty_primary`

### **HCP adoption — optum_hcp** (target + leaky-as-index dates)
**`adoption_status`** (target source; ≈2.3% ADOPTER) · `adoption_category` · `adoption_cumulative_share` · `adopter_rank` · `adopter_count` · **`first_adoption_dt`** (outcome event — leaky; null for 97.7%) · `days_to_first` · **`launch_dt`** (constant `2014-03-21`) · `launch_context` (`CSU_FDA_APPROVAL`) · `is_csu_approved` · `target_event_count` · `target_patient_count` · `distinct_target_code_count` · `medical_claim_count` · `medical_patient_count`

### HCP engagement — veeva_hcp (mostly empty for optum_hcp)
`engagement_score`(+`_100pt`/`_digital`/`_category`/`_method`/`_source`) · `visit_frequency_score` · `activity_completion_score` · `recency_score` · `tactic_execution_score` · `insight_engagement_score` · `key_message_score` · activity/visit counts · `last_activity_dt` · `days_since_last_activity` · `trigger_*` counts · `last_trigger_{delivery,acceptance}_ts` · `period_start`

### Market — market entity
`brand_trx` · `market_trx` · `market_share`(+`_pct`) · `brand/market_{patient,hcp}_count` · `brand_ndc_count` · `market_brand_count` · `business_metric_name` · `market_share_method` · `market_basket_definition` · `authoritative_source_note`

### Lineage / provenance (all grains)
`min/max_source_timestamp` · `min/max_ingestion_timestamp` · `min/max_data_lag_hours` · `source_tables_used` (array) · `extract_ym_values` (array) · `versions` (array). **These are ETL timestamps, not clinical dates** — `min/max_source_timestamp` sit entirely in 2025-09-30…2025-12-31 (the extract months) while `index_date` spans 2016–2024 (`min_source < index` share = 0.0).

---

## Comorbidity lookback window — certification status

**Question:** are the patient `cci_*` / `elx_*` / `charlson` / comorbidity-count columns computed strictly **pre-index**, or could the upstream window bleed past `index_date` (silent post-index leakage)?

**Evidence gathered (all on the real mart):**

1. **Source family** — `comorbidity_source_tables` is dominated by `{diagdemo, zip5_diag}` (+ occasional zip5 referral rollups). Comorbidities derive from **diagnosis** source tables. (`source_tables_used` per patient = diagnosis ± inpatient/lab.)
2. **ETL timestamps don't help** — `min/max_source_timestamp` are extract timestamps (2025), not clinical service dates, so they cannot bound the clinical window.
3. **Empirical no-scaling probe** (814,587 patient rows) — comorbidity counts/scores correlate ≈ **−0.03** with post-index span and ≈ **+0.02…+0.05** with pre-index span → they do **not** grow with observation length in either direction → signature of a **fixed pre-index lookback**, not an open window. (Reproducible probe in the findings report.)
4. **Our reference implementation is strictly pre-index** — `scripts/convert_optum_rwd.py` computes the same Charlson/Elixhauser families over `(index − LOOKBACK_DAYS, index]` with `LOOKBACK_DAYS = 180` (`:68`, gate `:1208`), per analyst-spec §4 ("180d lookback"). The production enrollment regime is `pre_days=360 / post_days=180` (`:117`).

**Verdict:** **Strong, convergent evidence that the comorbidity window is pre-index** (empirical no-scaling + our own strict-pre-index reference + spec intent). **Not yet certified**, because:

- A correlation test rules out a window that *scales* with enrollment but **cannot** rule out a *fixed-width* post-index bleed (e.g., `[index−365, index+90]` captures a constant post-index slice).
- The mart exposes **no per-diagnosis service date**, so the window cannot be measured from the file alone.

**To certify (the one external artifact still needed):** request, from the team that owns the **Spark mart-build pipeline v9.1**, the **comorbidity feature spec** — specifically the diagnosis lookback window definition relative to `index_date` (and whether any post-index diagnoses contribute). That single confirmation flips this from "strongly evidenced pre-index" to "certified."

---

## Pointers
- `docs/data/optum_mart_column_schema.csv` — exhaustive 252-column name+type list.
- `docs/reports/optum-mart-data-treatment-findings-20260608.md` — the temporal data-treatment audit (this dictionary closes its open "vendor data dictionary" item as far as the file allows).
- `docs/OPTUM_MART_CONVERSION.md` — the mart adapter + comorbidity-opacity caveat.
- `scripts/convert_optum_rwd.py` — reference comorbidity/window implementation (`LOOKBACK_DAYS=180`, strict pre-index).
- `src/data/manifests/optum_mart_feature_manifest.py` — the 64-col tier-0 safe allow-list + `knowable_at` contracts.
