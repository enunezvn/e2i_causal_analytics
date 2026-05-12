# Optum RWD Conversion

Converts the Optum claims parquet drop in `data/rwd/Optum_Parquet/` into
canonical per-cohort E2I parquet that the Tier-0 pipeline consumes identically
to synthetic data. Implements the leakage-safe cohort shaping specified in
`.claude/plans/csu-rwd-analyst-spec.md` (§3-§8).

Paired with the domain-agnostic file-ingestion capability added to
`data_preparer` (see `CONTRACT_VALIDATION.md`) and the shared RWD helpers in
`scripts/rwd_common.py`.

## When to use

- You have a new Optum-shaped parquet drop and need tier-0-ready cohorts.
- You need a leakage-safe CSU biologic cohort (initiation / discontinuation /
  persistence) without depending on the vendor-assigned `indexdt` field.
- You want to regenerate cohorts after receiving an updated Optum extract.

For CSU Excel data, use `scripts/convert_csu_rwd.py` instead.

## Inputs

Expected files under `data/rwd/Optum_Parquet/`:

| File | Role |
|------|------|
| `demographics.parquet` | Patient demographics, eligibility dates, single diagcode |
| `medication.parquet` | Prescription fills (NDC, HCPCS, days_supply, NPI) |
| `procedure.parquet` | Procedure claims (CPT/HCPCS, NPI) |
| `lab.parquet` | Lab results (LOINC, rslt_nbr, abnl_cd) — ~850K rows |
| `inpatientdata.parquet` | Inpatient admissions with diag1..5, proc1..5, admit/disch dates |
| `provider.parquet` | Provider taxonomy for specialty inference |

The directory is git-ignored — drop new extracts in place without committing.

## Outputs

For each built cohort, the converter writes:

```
data/rwd/optum/<cohort>/
  e2i_ml_v3_patient_journeys.parquet    # primary — one row per kept patient
  e2i_ml_v3_treatment_events.parquet    # lookback-window events
  e2i_ml_v3_hcp_profiles.parquet        # providers for kept patients
  e2i_ml_v3_split_registry.json         # chronological split config
  data_dictionary.csv                    # per-feature provenance + justification
  attrition_report.csv                   # patient counts at each filter step
```

`patient_journeys.parquet` carries a precomputed `data_split` column
(`train`/`validation`/`test`/`holdout`) so the tier-0 data_loader honors the
chronological split verbatim, no re-splitting. All cohort-specific targets
(`initiated_biologic_180d`, `discontinued_180d`, `persistent_at_180d`) are
stored as columns; the tier-0 runner selects the right one per cohort.

## Cohorts

| Cohort | Population | Index Date | Target | Output Dir |
|--------|-----------|-----------|--------|------------|
| **initiation** (A) | Treatment-naïve CSU patients | Diagnosis-anchored (§3.2) — inpatient L50.x admit, or first claim-dated event (fallback) | `initiated_biologic_180d` — any Xolair/Dupixent fill in `[index, index+180d]` | `data/rwd/optum/initiation/` |
| **discontinuation** (B) | Patients with ≥1 biologic fill | Re-anchored to first biologic fill date | `discontinued_180d` — gap > 90d between (last fill end) and next fill within 180d of initiation | `data/rwd/optum/discontinuation/` |
| **persistence** (C) | Same as B | Same as B | `persistent_at_180d` — active fill covering day 180 (days_supply-based, no gap > 60d) | `data/rwd/optum/persistence/` |

Each cohort is a **separate population** with its own index date — do not
assume a patient appearing in cohort A also appears in B/C.

## Usage

```bash
# Full run — all three cohorts
python scripts/convert_optum_rwd.py

# Single cohort
python scripts/convert_optum_rwd.py --cohort initiation

# Pilot (small subset, useful for dev loops)
python scripts/convert_optum_rwd.py --max-patients 500 --output /tmp/optum_pilot

# Dry-run (read + clean only, no writes)
python scripts/convert_optum_rwd.py --dry-run --verbose

# With §11 pilot audit (requires src.agents.ml_foundation.data_preparer.nodes.leakage_detector importable)
python scripts/convert_optum_rwd.py --max-patients 500 --pilot-audit
```

CLI flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--input DIR` | `data/rwd/Optum_Parquet/` | Parquet source directory |
| `--output DIR` | `data/rwd/optum/` | Root for per-cohort output subdirs |
| `--cohort {initiation,discontinuation,persistence,all}` | `all` | Which cohorts to build |
| `--max-patients N` | all | Limit to first N demographics rows (pilot / CI speedup) |
| `--pilot-audit` | off | Run `leakage_detector` on output as a §11 go/no-go gate |
| `--dry-run` | off | Load + clean only, no cohort build, no writes |
| `--verbose` | off | DEBUG-level logging |

## Index-date derivation (spec §3.2)

The spec's non-negotiable rule is: **never anchor the index date on the
vendor-assigned `indexdt` field**. Vendor `indexdt` is suspected of being
defined based on treatment status, which would leak the target into every
lookback feature.

The converter derives Cohort A index date in priority order:

1. **≥2 distinct L50.x inpatient claims** → use the 2nd admit date (§3.2 rule 1).
2. **Exactly 1 L50.x inpatient claim** → use that admit date (§3.2 rule 2).
3. **Pragmatic fallback** (documented): earliest claim-dated event in
   `medication`/`procedure`/`lab` that falls within
   `[demographics.eligeff, demographics.eligend]`, for patients whose
   demographics `diagcode` is L50.x. This is a conservative interpretation of
   "use the first qualifying claim date" when inpatient corroboration is
   unavailable. The anchor is never the vendor `indexdt` — it is always an
   observed claim.

Patients with only a demographics-level L50.x diagcode and no corroborating
claim of any kind (i.e., no claims at all) are dropped — the spec's
leakage-safe default.

For Cohort B/C: index is re-anchored to the first CSU biologic fill date
(`Brand_Name ∈ {XOLAIR, DUPIXENT}` or matching NDC prefix / HCPCS).

## Feature catalog (spec §7)

All features are computed in the **lookback window `[index_date − 180,
index_date − 1]`** — disjoint from the prediction window where targets live.

| Group | Spec ref | Examples |
|-------|----------|----------|
| Demographics (at index) | §7.1 | `age_at_index`, `gender`, `zip3`, `geographic_region`, `insurance_product`, `plan_type`, `urban_rural_code` |
| CSU disease characteristics | §7.2 | `dx_l50_1_count`, `dx_l50_8_count`, `dx_l50_9_count`, `dx_angioedema_count`, `csu_chronicity` |
| Comorbidity burden | §7.3 | `has_<cond>` + `<cond>_claim_count` per comorbidity in §6.3; `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score` |
| Healthcare utilization | §7.4 | `office_visits_total`, `office_visits_allergist`, `office_visits_dermatology`, `ed_visits_*`, `hospitalizations_total`, `unique_providers` |
| Non-target drug exposure | §7.5 | Per drug class (H1 1g/2g, H2, LTRA, steroids, immunosupp): `<class>_ever_filled`, `<class>_fill_count`, `<class>_days_supply_total`, `<class>_days_since_last_fill` |
| Labs | §7.6 | Per LOINC: `<lab>_tested`, `<lab>_result_last`, `<lab>_abnormal_flag` |
| Provider mix | §7.7 | `primary_specialist_type` (taxonomy1), `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_concentration` (HHI) |

**Critical anti-leakage rule** (§7.5): biologic fills (Xolair/Dupixent) are
explicitly **excluded** from non-target drug class features via
`_csu_biologic_mask`, preventing target leakage into the features.

See the `data_dictionary.csv` in each cohort output directory for the
complete, self-documenting list with source-table provenance and null rates.

## Known approximations

| Area | Approximation | Why |
|------|---------------|-----|
| `age_at_index` | Integer age from `demographics.age` (not exact DOB) | Optum provides `age`/`birth_yr` only |
| `urban_rural_code` | Minimal zip3→{urban,suburban} crosswalk | A full RUCA crosswalk requires a separate reference table — TODO |
| `elixhauser_score`, `charlson_score` | Default path uses Quan (2005) ICD-10 mappings with Charlson weights (1/2/3/6) and van Walraven (2009) Elixhauser weights. Legacy approximations retained behind `comorbidity_method="approx"` for parity testing. See issue #156 item 3. | Validated mappings replace prior chapter-count / high-severity proxies as of v4.2+. |
| Non-inpatient dx codes | Demographics single `diagcode` used as a proxy for baseline condition presence | Optum parquet has no claim-level dx outside inpatient diag1..5 |
| `source_timestamp` (issue #155 §3) | LAST_DAY of `extract_ym` month at 23:59:59 UTC | Optum vendor drops carry month granularity only; using LAST_DAY is the worst-case estimate and NEVER understates lag. Off by up to 30 days. |
| `adoption_category` Dupixent-CSU (issue #155 §1) | Dupixent CSU fills BEFORE 2025-04-18 flagged `dupixent_offlabel=TRUE` and EXCLUDED from Rogers diffusion curve; fills on/after are on-label and counted in the unified CSU curve (anchored at Xolair launch 2014-03-21) | FDA approved Dupixent for CSU (adults + adolescents ≥12y) on 2025-04-18 (Sanofi press release; FDA label 761055s070). Pre-approval fills would skew Rogers ranks; post-approval fills are valid on-label adoption. |
| `journey_stage` (issue #155 §2) | `prescribed` value NOT emitted from Optum converter | Optum claims are dispensed-only; no Rx-written signal. Reserved for cohorts with EHR Rx streams. |

These are documented per-feature in `data_dictionary.csv`.

## Drug-class-aware gap thresholds (issue #156 item 7)

Discontinuation and persistence detection use class-specific gap thresholds
via the `GAP_THRESHOLDS` dict in `scripts/convert_optum_rwd.py`. CSU biologics
(Xolair/Dupixent) use the historical `biologic` entry (90-day discontinuation,
60-day persistence) — behavior is bit-for-bit unchanged for the CSU cohort.

| Class | Discontinuation (days) | Persistence (days) |
|-------|------------------------|--------------------|
| `biologic` | 90 | 60 |
| `oral_chronic` | 60 | 30 |
| `specialty_injectable` | 90 | 60 |
| `default` | 60 | 30 |

When the converter is extended to non-biologic chronic therapies (e.g. CSU
antihistamine adherence, immunosuppressants), the class label is resolved
from `NON_TARGET_DRUG_CLASSES` at scoring time.

## Weighted data_quality_score (issue #156 item 4)

`data_quality_score` is computed per-claim and averaged per-patient over the
lookback window. Weights sum to 1.0:

```
claim_dqs = 0.40 * dx_complete
          + 0.25 * proc_complete
          + 0.20 * cost_complete
          + 0.15 * enroll_complete
patient_dqs = mean(claim_dqs over all claims in lookback)
```

Component rules:

- `dx_complete` = 1 if any of `diag1..5` (inpatient) or `diagcode` (demographics)
  is non-null and not `UNK`, else 0.
- `proc_complete` = 1 if `proc_code` (CPT/HCPCS) is non-null, else 0.
- `cost_complete` = 1 if `std_cost` is present; 0.5 if `std_cost` is null but
  any of `charge`/`copay`/`coins`/`deduct` is present; 0 otherwise. Medical
  claims are NOT penalized for missing `dispfee` / `avgwhlsl` (pharmacy-only).
- `enroll_complete` = 1 if both eligibility dates are non-null AND
  `continuous_enrollment == 1`; 0.5 if dates present but `continuous_enrollment != 1`;
  0 if any date null.

Patients with zero claims in the lookback window fall back to a feature-
completeness fraction so they still receive a non-null DQS (cohort eligibility
is gated elsewhere). The four payer-audit raw fields are excluded from this
fallback to preserve pre-PR DQS values.

## Soft enrollment filter (issue #156 item 5)

Opt-in via `--soft-enrollment-filter` CLI flag (default off — strict-mode
behavior preserved bit-for-bit). When enabled:

- The hard `continuous_enrollment == 1` gate is bypassed.
- `_check_enrollment_window` accepts any non-null eligibility span.
- Partial-enrollment patients receive a lower DQS via `enroll_complete < 1.0`.
- `--min-data-quality-score` threshold (default 0.50) is logged in
  `attrition_report.csv` under `soft-filtered (low DQS)` — patients are NOT
  dropped at ETL time; analysts choose the cutoff at model-training time.

## payer_category 8-vocabulary (issue #156 item 6)

`payer_category` extends the legacy 3-way `insurance_type` mapping with an
8-value vocabulary derived from `(bus, product, health_exch, lis_dual)` per
the priority rules in `scripts/rwd_common.derive_payer_category`:

| Vocab | Trigger |
|---|---|
| `commercial_exchange` | bus=COM AND health_exch |
| `commercial` | bus=COM |
| `medicare_lis_dual` | bus=MCR AND lis_dual |
| `medicare_advantage` | bus=MCR AND product∈{MA,MAPD} |
| `medicare` | bus=MCR |
| `medicaid` | bus=MCD |
| `cash` | bus=CASH |
| `other` | anything else |

The four raw source fields (`payer_bus_raw`, `payer_product_raw`,
`payer_health_exch_raw`, `payer_lis_dual_raw`) are persisted alongside the
derived value for re-derivation without re-ETL. Legacy `insurance_type` is
preserved for back-compat (deprecation in a future PR). Specialty-pharmacy
sub-vocabulary is out of scope here — requires NPPES taxonomy lookup
(tracked in #154).

Schema migration: `database/migrations/036_add_payer_category.sql`
(forward-only; CHECK constraint on the 8 values; partial index on
`payer_category IS NOT NULL`).

## Attrition expectations

Given the Optum parquet's fragmented patient panels (5,000 demographics rows,
only 145 with medication fills, ~10 with inpatient L50.x), spec-compliant
filtering produces smaller cohorts than a claims warehouse would.
Representative numbers from a 500-patient pilot:

```
initiation: start                                    500
initiation: age 18-89                                344
initiation: continuous_enrollment=1                  344
initiation: L50.x diagcode present                   315
initiation: after index + enrollment + exclusions     95
initiation: journeys constructed                      95
```

Full 5,000-patient runs scale proportionally. If a cohort drops below
`OptumTestConfig.min_eligible_patients = 30`, the tier-0 runner surfaces a
clear error and blocks pipeline execution — that is the desired behavior.

The per-cohort `attrition_report.csv` is the authoritative record of drop-off
at every filter step and should be reviewed after each conversion run.

## Running the Tier-0 pipeline

After conversion, the Tier-0 pipeline runs against each cohort via the
dedicated runner:

```bash
# Initiation cohort (primary)
python scripts/run_optum_tier0_test.py --cohort initiation

# Specific step for debugging
python scripts/run_optum_tier0_test.py --cohort initiation --step 2

# Discontinuation / persistence cohorts
python scripts/run_optum_tier0_test.py --cohort discontinuation
python scripts/run_optum_tier0_test.py --cohort persistence
```

The runner sets `OptumTestConfig.min_auc_threshold = 0.65` (higher than CSU
V1's 0.55) on the assumption that leakage-safe V2-style shaping produces
cleaner feature-target relationships.

The data_preparer loads the cohort parquet directly — no JSON conversion
needed — via the generic `FileIngestor` capability. The precomputed
`data_split` column is honored; no re-splitting is applied.

## Related files

- `scripts/convert_optum_rwd.py` — the converter itself
- `scripts/run_optum_tier0_test.py` — tier-0 runner for cohort outputs
- `scripts/rwd_common.py` — shared RWD helpers (IDs, regions, splitter, writers)
- `src/agents/ml_foundation/data_preparer/ingestion/` — generic file ingestion
- `src/agents/ml_foundation/data_preparer/CONTRACT_VALIDATION.md` — documents
  the `data_source` shape used to trigger file-based ingestion
- `.claude/plans/csu-rwd-analyst-spec.md` — the analyst spec (§3–§8 cohort
  definitions, exclusion lists, feature catalog, target derivations)
- `.claude/plans/optum-rwd-ingestion.md` — the implementation plan this work
  executes

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Cannot write struct type 'lab_values'` on parquet write | Empty nested dict in records | Handled — `_normalise_events_for_parquet` JSON-encodes `lab_values` |
| `Cohort X has 0 eligible patients` | Input data lacks claim-dated L50.x events for anyone meeting enrollment | Verify demographics `diagcode` coverage and `continuous_enrollment == 1` counts. Pilot-audit a larger `--max-patients` |
| Tier-0 runner errors "RWD data not found" | Converter not run, or wrong `--cohort` subdir | Run `python scripts/convert_optum_rwd.py --cohort <name>` first |
| Very low positive rate on `initiated_biologic_180d` | Normal for Optum V1-shape parquet (only 145/5000 patients have any meds) | Expected — documented in `attrition_report.csv`. Cohort B/C will be even smaller as subsets of A positives |
| MyPy complains about `data_source: str \| dict[str, Any]` | You passed a dict through a Supabase-only code path | The file-ingestion path is dispatched before the table-name path in `data_loader.py` — ensure `scope_spec.data_source` has `"type"` set to `"file_dir"` or `"files"` |
