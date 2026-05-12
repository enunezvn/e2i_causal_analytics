# CSU RWD Per-Field Lineage Audit

**Audit date:** 2026-05-04
**Audited file:** `scripts/convert_csu_rwd.py` (HEAD `102ec43`)
**Audit scope:** Every output column written to the four JSON files emitted by `CSUDataConverter.write_all()`:

- `e2i_ml_v3_hcp_profiles.json`
- `e2i_ml_v3_patient_journeys.json`
- `e2i_ml_v3_treatment_events.json`
- `e2i_ml_v3_split_registry.json`

This is a code-reading audit. The vendor Excel workbook `data/rwd/csu/csu_data.xlsx` is gitignored; columns marked "raw Excel column" are inferred from the converter's read paths and the analyst spec at `.claude/plans/csu-rwd-analyst-spec.md`. Re-validation against the workbook (when available) is a follow-up TODO and would only refine UNKNOWN rows — none of the OBSERVABLE / POST-INDEX verdicts depend on raw-data inspection.

---

## 1. Lookback contract

The downstream ML-foundation pipeline expects all features to be observable in the lookback window:

```
lookback window = [index_date − 180d, index_date − 1d]
```

- **Source of truth for the contract:** `config/cohort_vocabulary.yaml` (`baseline_period_days: 180`, `washout_period_days: 30`, `followup_period_days: 365`) and `.claude/plans/csu-rwd-analyst-spec.md` §4 ("Temporal Architecture").
- **What "OBSERVABLE" means here:** The field's value at output reflects only data points whose timestamps fall in (or before) the lookback window — i.e. the field could have been written down on day `index − 1` without referring to any future event.
- **What "POST-INDEX" means:** The field aggregates data from on or after `index_date`, OR is computed from a target-equivalent quantity (e.g. counts derived from medication fills when `treatment_initiated` is itself defined as "patient appears in medication sheet"). These fields constitute structural leakage even before any temporal join.
- **What "UNKNOWN" means:** The converter logic does not anchor the field to a temporal window, AND the field's value can be either pre- or post-index depending on raw-data semantics that aren't explicit in code (e.g. aggregate counts whose date filter is the whole patient panel, or vendor-assigned IDs whose source date is unrecorded).

The current `convert_csu_rwd.py` does **NOT** apply a lookback window to its aggregates — the analyst spec was written specifically because the converter ignores temporal alignment. Rows below mark POST-INDEX where the absence of a window combined with the field's semantics produces structural leakage; UNKNOWN where the absence of a window leaves the verdict ambiguous.

---

## 2. ML-foundation consumers

For the consumer column, "Yes" means at least one occurrence of the field name appears in `src/agents/ml_foundation/`. Generic field names (`brand`, `patient_id`, `data_split`) appear pervasively and are noted as "Yes (pervasive)". `disease_severity`, `engagement_score`, `days_on_therapy`, `hcp_visits` are explicitly named in `src/agents/ml_foundation/data_preparer/nodes/leakage_detector.py` (lines 582, 1000) and `feature_analyzer/nodes/feature_generator.py` (lines 431-466) — the leakage detector knows about them precisely because they are the canonical leakage exemplars.

---

## 3. Patient journeys (`e2i_ml_v3_patient_journeys.json`)

Built by `_build_patient_journeys()` (lines 587-870). One record per patient in the master registry.

| # | Output column | Source column | Transformation | Nullability | Temporal alignment | Consumer |
|---|---|---|---|---|---|---|
| 1 | `patient_journey_id` | derived from `patid` sequence | `f"PJ_{seq:06d}"` (assigned in `_build_patient_id_map`) | not-null | OBSERVABLE (synthetic ID, no temporal content) | Yes (pervasive) |
| 2 | `patient_id` | derived from `patid` sequence | `f"PAT_{seq:06d}"` | not-null | OBSERVABLE (synthetic ID) | Yes (pervasive) |
| 3 | `patient_hash` | `demo.patid` | SHA-256 of patid → first 20 hex chars | not-null | OBSERVABLE | No |
| 4 | `journey_start_date` | `demo.indexdt` (or earliest clinical date if missing) | `_safe_date(index_date)` | nullable | OBSERVABLE (this *is* the index anchor; vendor-assigned, see §6) | Yes |
| 5 | `journey_end_date` | latest of (`demo.eligend`, last `med.medication_date + days_sup`, `proc.proc_date.max()`, `lab.fst_dt.max()`) | `_safe_date(end_date)` | nullable | **POST-INDEX** — derived from latest available clinical event (post-index by construction) | Yes |
| 6 | `journey_duration_days` | derived (end − start) | integer days | nullable | **POST-INDEX** (depends on `journey_end_date`) | Yes |
| 7 | `journey_stage` | derived from `treatment_initiated` + `days_on_therapy` | enum: `treatment_optimization` / `initial_treatment` / `diagnosis` | not-null | **POST-INDEX** — depends on POST-INDEX inputs | Yes |
| 8 | `journey_status` | derived from `discontinuation_flag` + `treatment_initiated` | enum: `completed` / `active` / `monitoring` | not-null | **POST-INDEX** — same | Yes |
| 9 | `primary_diagnosis_code` | `demo.diagcode` | `_format_diagcode()` (insert dot, default `L50.8`) | not-null (defaults `L50.8`) | OBSERVABLE (single dx code per patient; vendor-assigned, no claim date) | Yes |
| 10 | `primary_diagnosis_desc` | constant `"Chronic Spontaneous Urticaria"` | hard-coded | not-null | OBSERVABLE | No |
| 11 | `secondary_diagnosis_codes` | constant `[]` | hard-coded empty list | not-null (empty) | OBSERVABLE | No |
| 12 | `brand` | derived from `treatment_initiated` | `"competitor"` if treated else `None` | nullable | **POST-INDEX** — definitionally equivalent to target | Yes (pervasive) |
| 13 | `age_group` | `demo.age` | binned via `_age_group()` | nullable | OBSERVABLE (age at vendor's index assignment) | Yes |
| 14 | `gender` | `demo.gdr_cd` | normalised to `F`/`M`/`None` | nullable | OBSERVABLE | Yes |
| 15 | `geographic_region` | `demo.zipcode_5` | `_map_zipcode_to_region()` (Census 3-digit ZIP) | nullable | OBSERVABLE | Yes |
| 16 | `state` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 17 | `zip_code` | `demo.zipcode_5` | first ZIP if multi-ZIP underscore-joined | nullable | OBSERVABLE | No |
| 18 | `insurance_type` | `demo.bus` | `_insurance_type()` (COM/MCR/MCD/Other) | nullable | OBSERVABLE | Yes |
| 19 | `data_quality_score` | derived from archetype | random uniform in archetype-dependent band | not-null | OBSERVABLE (synthetic noise, not informative) | Yes |
| 20 | `comorbidities` | constant `[]` | hard-coded empty list | not-null (empty) | OBSERVABLE | No |
| 21 | `risk_score` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 22 | `data_source` | constant `"RWD_Claims"` | hard-coded | not-null | OBSERVABLE | No |
| 23 | `data_sources_matched` | constant `["RWD_Claims"]` | hard-coded | not-null | OBSERVABLE | No |
| 24 | `source_match_confidence` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 25 | `source_stacking_flag` | constant `False` | hard-coded | not-null | OBSERVABLE | No |
| 26 | `source_combination_method` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 27 | `source_timestamp` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 28 | `ingestion_timestamp` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE (run-time clock; not a feature) | No |
| 29 | `data_lag_hours` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 30 | `data_split` | derived from chronological sort of `journey_start_date` | enum: `train`/`validation`/`test`/`holdout` | nullable initially → not-null after `_apply_chronological_split` | OBSERVABLE (split label, not a feature) | Yes (pervasive) |
| 31 | `split_config_id` | `uuid.uuid4()` per converter run | string UUID | not-null | OBSERVABLE | Yes |
| 32 | `created_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |
| 33 | `updated_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |
| 34 | `treatment_initiated` | derived from `_med_by_pat` membership | `1` if patient appears in medication sheet, else `0` | not-null | **POST-INDEX** — this is the **target**; "patient has any med fill, ever". See gap report §3 root cause #2. | Yes (pervasive — target) |
| 35 | `discontinuation_flag` | derived from `_derive_discontinuation_flag()` | `1` if med-fill gap > 90d (or > 90d since last fill end), else `0` for medicated; `None` for unmedicated | nullable | **POST-INDEX** — gap analysis spans entire med history, post-index by construction | Yes (target B) |
| 36 | `disease_severity` | derived from `_derive_disease_severity()` | aggregate of L50.x dx + med fill count + J2357 proc count + abnormal lab flag (no time filter) | not-null | **POST-INDEX** — counts span entire patient panel; one of the 5 known-leaky features (single-feature AUC ≥ 0.99 per Apr 12 runs; cited by leakage_detector.py:1000 as exemplar) | Yes (named exemplar in leakage detector) |
| 37 | `engagement_score` | derived from `_derive_engagement_score()` | aggregate of unique HCPs + med fill count + lab count + continuous_enrollment flag (no time filter) | not-null | **POST-INDEX** — same; one of the 5 known-leaky features | Yes (named in `scope_definer/nodes/scope_builder.py:250`, `feature_generator.py:466`) |
| 38 | `days_on_therapy` | sum of `med.days_sup` for all fills (no time filter) | integer | not-null (`0` if not in med sheet) | **POST-INDEX** — sums over entire med panel; one of the 5 known-leaky features (cited by leakage_detector.py:582) | Yes (named in `feature_generator.py:431`) |
| 39 | `hcp_visits` | unique `(npi, medication_date)` pairs in med (no time filter) | integer | not-null (`0` if not in med sheet) | **POST-INDEX** — counts span entire med panel; one of the 5 known-leaky features | Yes (named in `feature_generator.py:431`) |
| 40 | `prior_treatments` | distinct `brand_normalised` in med where `medication_date < index_date` | integer | not-null (`0` if not in med sheet) | OBSERVABLE — explicit `< index_date` filter; this is the only field with a real lookback gate | Yes |
| 41 | `age_continuous` | `demo.age` | float (raw, not binned) | nullable | OBSERVABLE | No |
| 42 | `eligibility_duration_days` | `demo.eligend − demo.eligeff` | integer days, clipped to ≥ 0 | nullable | UNKNOWN — `eligend` extends past index by design (active enrollment); duration overlaps both pre- and post-index periods. Treat as POST-INDEX in practice unless masked to pre-index portion only. | No |
| 43 | `medication_claim_count` | `len(_med_by_pat[patid])` (no time filter) | integer | not-null (`0` if not in med sheet) | **POST-INDEX** — counts entire med panel; one of the 5 known-leaky features (per gap report; behaves identically to `days_on_therapy` and `hcp_visits`) | No (but trivially equivalent to `treatment_initiated > 0`) |
| 44 | `procedure_claim_count` | `len(_proc_by_pat[patid])` (no time filter) | integer | not-null (`0` if not in proc sheet) | **POST-INDEX** — counts entire proc panel | No |
| 45 | `lab_claim_count` | `len(_lab_by_pat[patid])` (no time filter) | integer | not-null (`0` if not in lab sheet) | **POST-INDEX** — counts entire lab panel | No |
| 46 | `demo_<col>` (dynamic pass-through) | any `demo` column not in the explicit hard-coded set | stringified value | nullable | UNKNOWN — depends on the upstream column; safe iff column is purely demographic | No |

### 3.1 Notes on the patient-journey table

- **Index-date semantics.** `journey_start_date` is the vendor-assigned `demo.indexdt` if present, else the earliest clinical event date. The analyst spec §3 explicitly warns this is **not a usable lookback anchor for ML** because (a) demo-only patients have no observable index, (b) clinical-only patients use their first claim date which is itself part of the medication panel that defines the target. The converter writes the field anyway for downstream pipeline compatibility.
- **The 5 known-leaky features.** Rows 36 (`disease_severity`), 37 (`engagement_score`), 38 (`days_on_therapy`), 39 (`hcp_visits`), 43 (`medication_claim_count`) are the canonical CSU leakage set documented in the gap report (`docs/results/rwd_pipeline_run_20260412_*.md`, single-feature AUC ≥ 0.99). All five are POST-INDEX because the converter does not apply a `< index_date` filter to the underlying aggregates. `prior_treatments` (row 40) demonstrates the masking would be a one-line change per field — the temporal predicate is already used there.
- **`prior_treatments` is the OBSERVABLE proof-of-concept.** It is the single field in the entire converter that filters by `medication_date < index_date`. If the masking work proceeds, the same predicate can be lifted onto rows 36–39 and 43 (also row 8 `journey_status`, which depends on those).
- **Backlog #17 reconciliation (2026-05-12).** Rows 36 (`disease_severity`), 37 (`engagement_score`), 38 (`days_on_therapy`), 39 (`hcp_visits`), 40 (`prior_treatments`), and 43 (`medication_claim_count`) are now declared `KnowableAt(reference="post_index")` in `src/data/manifests/csu_feature_manifest.py`. This reconciles the manifest with this audit document's verdicts. The iter-5 empirical audit (2026-05-09) on real CSU n=9607 confirmed that even with `--lookback-days=180` applied at the converter, the six features remain target-coupled at Layer 3 z=14.13–69.18 — because the target `treatment_initiated` is defined as "patient appears anywhere in `_med_by_pat`", and untreated patients are absent from `_med_by_pat` entirely, so every medication-derived aggregate (count / sum / nunique) collapses to zero for them regardless of date windowing. The target-coupling is structural, not date-dependent. `prior_treatments` is included despite its `< index_date` filter (the predicate prunes events, but the underlying frame is still indexed by med-panel membership). Reclassifying the six to `post_index` moves the catch from Layer 3 (statistical, slower) to Layer 1 (declarative, deterministic, cheaper); the three dependent B3 engineered features (`engagement_per_visit`, `treatment_diversity_intensity`, `severity_engagement_product`) follow by chain validity. `procedure_claim_count` and `lab_claim_count` retain pre-anchor status because they derive from independent event panels (`_proc_by_pat` / `_lab_by_pat`).

---

## 4. HCP profiles (`e2i_ml_v3_hcp_profiles.json`)

Built by `_build_hcp_profiles()` (lines 469-581). One record per unique obfuscated NPI in the medication or procedure sheets.

| # | Output column | Source column | Transformation | Nullability | Temporal alignment | Consumer |
|---|---|---|---|---|---|---|
| 1 | `hcp_id` | derived from `npi` sort order | `f"HCP_{seq:06d}"` | not-null | OBSERVABLE (synthetic ID) | Yes |
| 2 | `npi` | `med.npi` (obfuscated) | `_generate_luhn_npi()` (SHA-256 → 9 digits + Luhn check) | not-null | OBSERVABLE | Yes |
| 3 | `first_name` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 4 | `last_name` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 5 | `specialty` | constant `"Allergy/Immunology"` | hard-coded | not-null | OBSERVABLE | Yes |
| 6 | `sub_specialty` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 7 | `practice_type` | derived from patient volume | `Hospital` if >100, `Group` if ≥50, else `Solo` | not-null | UNKNOWN — uses lifetime patient panel, not lookback | Yes |
| 8 | `practice_size` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 9 | `geographic_region` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 10 | `state` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 11 | `city` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 12 | `zip_code` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 13 | `priority_tier` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 14 | `decile` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 15 | `total_patient_volume` | unique patids per NPI in med + proc | integer count (no time filter) | not-null | UNKNOWN — lifetime panel; could span both pre- and post-index | Yes |
| 16 | `target_patient_volume` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 17 | `prescribing_volume` | row count per NPI in med + proc | integer count (no time filter) | not-null | UNKNOWN — lifetime panel | Yes |
| 18 | `years_experience` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 19 | `affiliation_primary` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 20 | `affiliation_secondary` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 21 | `digital_engagement_score` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 22 | `preferred_channel` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 23 | `last_interaction_date` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 24 | `interaction_frequency` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 25 | `influence_network_size` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 26 | `peer_influence_score` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 27 | `adoption_category` | derived from prescribing-volume quartile | enum: `innovator`/`early_adopter`/`early_majority`/`late_majority` | not-null | UNKNOWN — depends on lifetime `prescribing_volume`; stable HCP attribute but quartile boundaries shift with cohort | Yes |
| 28 | `coverage_status` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 29 | `territory_id` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 30 | `sales_rep_id` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 31 | `created_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE (run-time clock) | No |
| 32 | `updated_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |

### 4.1 Notes on HCP profiles

HCP profiles do not have a per-row index date — they are patient-population summaries. The volume columns (rows 7, 15, 17, 27) inherit any temporal leakage that the calling join introduces. If used as features in the patient-journey table, they should be re-aggregated against the lookback window at join time, not at converter-write time.

---

## 5. Treatment events (`e2i_ml_v3_treatment_events.json`)

Built by `_build_treatment_events()` (lines 876-1144). Four sub-types share the same schema; differences are in which columns are populated. Audit covers the schema (32 columns) once with sub-type notes inline.

| # | Output column | Source column | Transformation | Nullability | Temporal alignment | Consumer |
|---|---|---|---|---|---|---|
| 1 | `treatment_event_id` | derived | `f"TE_{te_seq:06d}"` | not-null | OBSERVABLE (synthetic ID) | Yes |
| 2 | `patient_journey_id` | derived from patid map | `PJ_{seq:06d}` | not-null | OBSERVABLE | Yes |
| 3 | `patient_id` | derived from patid map | `PAT_{seq:06d}` | not-null | OBSERVABLE | Yes |
| 4 | `hcp_id` | `med.npi` → `_hcp_npi_map` (only for prescription rows) | lookup; `None` for diagnosis/procedure/lab | nullable | OBSERVABLE for the join itself | Yes |
| 5 | `event_date` | per sub-type: `demo.indexdt` (diagnosis), `med.medication_date` (prescription), `proc.proc_date` (procedure), `lab.fst_dt` (lab_test) | `_safe_date()` | nullable | OBSERVABLE — this is the event's own timestamp | Yes |
| 6 | `event_type` | constant per source sheet | enum: `diagnosis`/`prescription`/`procedure`/`lab_test` | not-null | OBSERVABLE | Yes |
| 7 | `event_subtype` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 8 | `brand` | constant `"competitor"` for prescription, `None` otherwise | hard-coded | nullable | OBSERVABLE for the field itself; downstream joins must apply lookback | Yes |
| 9 | `drug_ndc` | `med.code` (prescription only) | `str(int(...))` | nullable | OBSERVABLE | Yes |
| 10 | `drug_name` | `med.brand_normalised` (prescription only) | normalised brand | nullable | OBSERVABLE | Yes |
| 11 | `drug_class` | constant `"Monoclonal Antibody"` for prescription, `None` otherwise | hard-coded | nullable | OBSERVABLE | No |
| 12 | `dosage` | `med.strength` (prescription only) | string | nullable | OBSERVABLE | No |
| 13 | `duration_days` | `med.days_sup` (prescription only) | `_safe_int()` | nullable | OBSERVABLE | No |
| 14 | `icd_codes` | `demo.diagcode` (diagnosis only) | `[_format_diagcode(...)]` | not-null (list, possibly empty) | OBSERVABLE | Yes |
| 15 | `cpt_codes` | `proc.proc_code` (procedure only) | `[str(...).strip()]` | not-null (list) | OBSERVABLE | Yes |
| 16 | `loinc_codes` | `lab.loinc_cd` (lab only) | `[str(...).strip()]` | not-null (list) | OBSERVABLE | Yes |
| 17 | `lab_values` | `lab.tst_desc` → `lab.rslt_nbr` (lab only) | dict `{tst_desc: rslt_nbr}` | not-null (dict, possibly empty) | OBSERVABLE | Yes |
| 18 | `location_type` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 19 | `facility_id` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 20 | `cost` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 21 | `outcome_indicator` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 22 | `adverse_event_flag` | constant `False` | hard-coded | not-null | OBSERVABLE | No |
| 23 | `discontinuation_flag` | constant `False` (event-level, distinct from journey-level) | hard-coded | not-null | OBSERVABLE | Yes |
| 24 | `discontinuation_reason` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 25 | `sequence_number` | derived per-patient counter | integer | not-null | OBSERVABLE (assignment order, not a feature) | No |
| 26 | `days_from_diagnosis` | `med.days_from_indexdt` (prescription, lab) or `(proc_date − indexdt).days` (procedure) or `0` (diagnosis) | integer | not-null | OBSERVABLE per row, but post-index when value > 0 — caller must apply lookback for safe joins | Yes |
| 27 | `previous_treatment` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 28 | `next_treatment` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 29 | `data_source` | constant `"RWD_Claims"` | hard-coded | not-null | OBSERVABLE | No |
| 30 | `source_timestamp` | constant `None` | hard-coded null | nullable | OBSERVABLE | No |
| 31 | `ingestion_timestamp` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |
| 32 | `data_split` | propagated from journey | enum | not-null after split | OBSERVABLE | Yes |
| 33 | `created_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |
| 34 | `updated_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |

### 5.1 Notes on treatment events

Treatment events are inherently OBSERVABLE row-by-row (each row carries its own `event_date`). The structural risk is in **downstream aggregation**: any code that reduces events to a per-patient feature without filtering by `event_date < index_date` recreates the same leakage as journey-level rows 36–39 and 43. The Optum converter handles this by emitting only events in `[lookback_start, index_date)` (`convert_optum_rwd.py:1164` — `win = grp[(grp["medication_date"] >= lb) & (grp["medication_date"] < idx)]`); the CSU converter does not pre-filter.

---

## 6. Split registry (`e2i_ml_v3_split_registry.json`)

Built by `_build_split_registry()` (lines 1227-1250). One record per converter run.

| # | Output column | Source column | Transformation | Nullability | Temporal alignment | Consumer |
|---|---|---|---|---|---|---|
| 1 | `split_config_id` | `uuid.uuid4()` | string UUID | not-null | OBSERVABLE | Yes |
| 2 | `config_name` | constant `"csu_rwd_v1"` | hard-coded | not-null | OBSERVABLE | No |
| 3 | `config_version` | constant `"1.0.0"` | hard-coded | not-null | OBSERVABLE | No |
| 4 | `train_ratio` | constant `0.60` | hard-coded | not-null | OBSERVABLE | No |
| 5 | `validation_ratio` | constant `0.20` | hard-coded | not-null | OBSERVABLE | No |
| 6 | `test_ratio` | constant `0.15` | hard-coded | not-null | OBSERVABLE | No |
| 7 | `holdout_ratio` | constant `0.05` | hard-coded | not-null | OBSERVABLE | No |
| 8 | `data_start_date` | min `journey_start_date` across all journeys | derived | nullable | OBSERVABLE | No |
| 9 | `data_end_date` | max `journey_start_date` | derived | nullable | OBSERVABLE | No |
| 10 | `train_end_date` | journey at index `train_end − 1` | derived | nullable | OBSERVABLE | No |
| 11 | `validation_end_date` | journey at index `val_end − 1` | derived | nullable | OBSERVABLE | No |
| 12 | `test_end_date` | journey at index `test_end − 1` | derived | nullable | OBSERVABLE | No |
| 13 | `temporal_gap_days` | constant `7` | hard-coded | not-null | OBSERVABLE | No |
| 14 | `patient_level_isolation` | constant `True` | hard-coded | not-null | OBSERVABLE | No |
| 15 | `split_strategy` | constant `"chronological"` | hard-coded | not-null | OBSERVABLE | No |
| 16 | `is_active` | constant `True` | hard-coded | not-null | OBSERVABLE | No |
| 17 | `created_at` | `datetime.now()` | `self.now_iso` | not-null | OBSERVABLE | No |

---

## 7. Audit summary

### 7.1 Counts

| Record type | Total columns | OBSERVABLE | POST-INDEX | UNKNOWN |
|---|---:|---:|---:|---:|
| Patient journeys | 46 | 35 | 10 | 1 + 1 dynamic-passthrough class (UNKNOWN) |
| HCP profiles | 32 | 28 | 0 | 4 |
| Treatment events | 34 | 34 | 0 | 0 |
| Split registry | 17 | 17 | 0 | 0 |
| **Grand total** | **129** | **114** (88%) | **10** (8%) | **5+ ** (4%) |

The 10 POST-INDEX entries are concentrated in **patient journeys**: rows 5, 6, 7, 8, 12, 34, 35, 36, 37, 38, 39, 43 (12 entries; row 35 `discontinuation_flag` is the secondary target so it's not "leaky" per se but post-index by definition). Counting `discontinuation_flag` and `treatment_initiated` as targets (not features) leaves 10 leaky-feature entries, which matches the 5 canonical leaky features (rows 36, 37, 38, 39, 43) plus 5 derived columns that depend on them (rows 5 `journey_end_date`, 6 `journey_duration_days`, 7 `journey_stage`, 8 `journey_status`, 12 `brand`).

### 7.2 The 5 known-leaky features — confirmation

Per the gap report (`docs/results/rwd_pipeline_run_20260412_*.md`), the leakage detector flagged the following with single-feature AUC ≥ 0.99:

| Feature | Audit row | POST-INDEX reason |
|---|---|---|
| `engagement_score` | §3 row 37 | Aggregates unique HCPs + med fills + lab counts over the entire patient panel; no `< index_date` predicate. Becomes deterministic when `treatment_initiated == 0` (zero med rows ⇒ engagement = 0 + 0 + lab/3) and tightly coupled when treated. |
| `days_on_therapy` | §3 row 38 | `int(med_df["days_sup"].fillna(0).sum())` over entire med panel. Strictly equals `0` for untreated patients and strictly `> 0` for treated, perfectly separating the target. |
| `hcp_visits` | §3 row 39 | Unique `(npi, medication_date)` pairs across entire med panel. Identical separation property to `days_on_therapy`. |
| `medication_claim_count` | §3 row 43 | `len(_med_by_pat[patid])` — literally the count that defines `treatment_initiated`. The only leak-safe value of this feature is `0`. |
| `disease_severity` | §3 row 36 | Adds `0.5 × med fills` (capped 3.0) and `0.5 × J2357 procs` (capped 2.0) and `+1.0` if any abnormal lab — all panel-wide. The `+0.5 × med fills` term alone makes `disease_severity > 2.0` essentially equivalent to `treatment_initiated == 1`. |

All five rows above are confirmed POST-INDEX by code-reading; no Excel-data inspection needed.

### 7.3 UNKNOWN entries — what would resolve them

| Field | Resolution path |
|---|---|
| `eligibility_duration_days` (journey row 42) | Mask to lookback portion only (`min(eligend, index_date) − eligeff`) — code change |
| `practice_type`, `total_patient_volume`, `prescribing_volume`, `adoption_category` (HCP rows 7, 15, 17, 27) | Re-compute against rolling-window panels at join time, or accept as stable HCP attributes (research literature treats prescribing volume as stable over multi-year horizons, so the practical leak risk is low if the index spread is < 12 months) |
| `demo_<col>` dynamic passthrough (journey row 46) | Per-column verification once the workbook is accessible; safe if columns are demographic constants (yob, ethnicity, etc.) |

---

## 8. Re-grade implications for §2 R3

### 8.1 Structural remediability verdict

**CSU is structurally remediable.** Five observations:

1. **Raw event timestamps exist.** The medication/procedure/lab sheets carry `medication_date`, `proc_date`, `fst_dt` — every aggregate that is currently POST-INDEX could be re-computed under a `< index_date` filter using the same columns the converter already reads.
2. **A working remediation pattern is already in the file.** `prior_treatments` (§3 row 40) demonstrates the exact pattern that needs to be applied to rows 36–39 and 43 — a one-line `event_date < index_date` predicate per aggregate.
3. **The Optum converter is the existence proof.** `convert_optum_rwd.py` implements the analyst spec's lookback architecture in production. Its `_compute_features()` builds 50+ pre-index features with explicit `[lb_start, lb_end]` filters. A masked CSU converter would converge on the same shape.
4. **However, the converter cannot rescue cohort comparability by itself.** Per analyst spec §3 root cause #1, only **196 of 9,607 (2%)** CSU patients have both demographics and clinical claims. Even with perfect masking, the demo-only patients have no observable lookback features by construction (their `_med_by_pat` / `_proc_by_pat` / `_lab_by_pat` entries are empty). Lookback masking eliminates the leakage but produces a feature matrix that is mostly zero for 70% of the cohort — which trains a classifier that learns "patient has clinical data ⇒ likely treated" instead of the original "patient has any med fill ⇒ definitely treated". The leakage-detector AUC will drop below 0.99 but the data quality remains questionable.
5. **The analyst spec's preferred fix is a re-pull, not a re-mask.** `.claude/plans/csu-rwd-analyst-spec.md` was written specifically because the current vendor file's fragmented panel structure cannot be repaired downstream. The spec defines three cohorts (initiation / discontinuation / persistence) with proper enrollment gates; this requires the analyst to re-query the warehouse, not patches in the converter.

### 8.2 Recommended R3 grade after this audit (PROVISIONAL)

The current §2 R3 grade is `D: ✅ | E: ✅ (synthetic) / ⚠️ (CSU RWD converter) | Ex: ⚠️ (no planted hazard suite)` (per `.claude/plans/tier0_evaluation_vs_distilled_mlops.md` line 105).

**Recommended update (PROVISIONAL pending masked-CSU re-run):**

```
R3 | Leakage prevention | D: ✅ | E: ✅ (synthetic) / ⚠️ (CSU RWD converter — confirmed POST-INDEX
                                                       on 5 features per per-field audit
                                                       2026-05-04, structural remediation
                                                       documented as scoped follow-up shard) | Ex: ⚠️
```

Rationale:
- **No upgrade.** The 5 leaky-feature confirmations are not new — the leakage detector has been catching them. The audit only formalises the lineage. Until either (a) the converter is masked + tier-0 re-runs green, or (b) the CSU vendor file is replaced with a re-pull conforming to the analyst spec, the E sub-grade for CSU stays ⚠️.
- **No downgrade.** The audit demonstrates that detection (D) continues to work and the *machinery* enforces the leakage gates. The E ⚠️ is a data-quality finding, not a pipeline-correctness finding.
- **Re-grade to ✅ for E (CSU)** would require: masked converter committed, `python scripts/run_tier0_test.py --data-dir data/rwd/csu` produces single-feature AUC < 0.85 on all numeric features, and the leakage detector's `LeakageFinding.severity == HIGH` count is < 3.

### 8.3 Suggested follow-up shard

A discrete, scoped shard would deliver the masking fix:

> **Shard:** `feat/phase4b-csu-converter-masking`
> **Scope:** Add a `--lookback-days` CLI flag (default 180) to `scripts/convert_csu_rwd.py`. Apply `event_date < index_date AND event_date >= index_date − lookback_days` to the aggregate computations behind `disease_severity`, `engagement_score`, `days_on_therapy`, `hcp_visits`, `medication_claim_count`, `procedure_claim_count`, `lab_claim_count`, and `eligibility_duration_days`. Mark the journey records with `journey_status: "lookback_masked"` so downstream code can detect mode.
> **Out of scope (separate work):** The fragmented-panel structural problem (§8.1 point 4); that requires the data re-pull described in the analyst spec, not a converter patch.
> **Acceptance:** `python scripts/run_tier0_test.py --data-dir data/rwd/csu` runs to completion without leakage detector flagging the 5 named features as CRITICAL, and produces a result doc at `docs/results/csu_post_lineage_audit_<ts>.md`.

This shard is gated on user approval per the close-out workflow; the present PR documents the lineage and leaves the executable masking as a clearly-scoped next step.
