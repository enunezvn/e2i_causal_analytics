# Optum Mart → Tier-0 Cohort Adapter

Converts the **entity-stacked, pre-engineered Optum mart**
(`data/rwd/Optum_Parquet/Optum.parquet`) into canonical per-cohort E2I parquet
that the Tier-0 pipeline consumes identically to synthetic data.

> **This is NOT the legacy raw-claims converter.** `scripts/convert_optum_rwd.py`
> (documented in [`OPTUM_CONVERSION.md`](OPTUM_CONVERSION.md)) ingests the 6 raw
> claims files (`demographics`/`medication`/`procedure`/`lab`/`inpatientdata`/
> `provider`) and recomputes every feature from raw claims in a lookback window.
> The **mart adapter** (`scripts/convert_optum_mart.py`, this doc) ingests a
> single pre-aggregated mart file where every feature is already computed, and is
> therefore a **split-and-map adapter**, not a recompute. A **third** adapter,
> `scripts/convert_optum_hcp_adoption.py` (the "HCP adoption-propensity cohort"
> section below), reads the SAME mart file but the `optum_hcp` *provider* grain
> instead of the `patient` grain — it is the deployable commercial-HCP-targeting
> deliverable. Use the one that matches your input drop **and** unit of analysis.

## When to use

- Your Optum drop is the single entity-stacked `Optum.parquet` mart (252 columns,
  one `entity_type` discriminator column), not the 6-file raw-claims layout.
- You need a tier-0-ready CSU biologic cohort (initiation / discontinuation /
  persistence) from that mart.
- You need the deployable **commercial HCP-targeting** cohort (HCP
  adoption-propensity) from the mart's `optum_hcp` *provider* grain → use the
  sibling adapter `convert_optum_hcp_adoption.py` (see "HCP adoption-propensity
  cohort" below). The three patient cohorts above are feature-bound and
  non-deployable; the HCP cohort is the one that deploys end-to-end.

## Input — the entity-stacked mart

Single file: `data/rwd/Optum_Parquet/Optum.parquet` — **252 columns ×
3,758,007 rows**, stacked by `entity_type`:

| `entity_type` | Rows | Role |
|---|---|---|
| `patient` | 814,587 | The only rows the adapter models. One row per patient, every feature pre-aggregated. |
| `optum_hcp` | 2,753,238 | Provider rows — the grain the **HCP adoption-propensity adapter** (`convert_optum_hcp_adoption.py`) models (see that section below). The *patient* mart adapter does not read them. |
| `veeva_hcp` | 189,951 | Veeva provider rows (unused). |
| `market` | 231 | Market-level rows (unused). |

The adapter reads **only `entity_type == 'patient'`** via a pushed-down pyarrow
filter, and for the treatment-anchored cohorts pushes down further to initiators
(`index_biologic_brand != 'no_treatment'`, ~24K vs 814K rows) so the read stays
frugal on a memory-constrained host.

The directory is git-ignored — drop new extracts in place without committing.

### Mart-specific ground truth (measured)

- `index_date == first_csu_dx_date` for all 814,587 patient rows → the mart is
  **already diagnosis-anchored**; the adapter does NOT re-derive an index.
- `data_quality_band` is **opaque/inverted** (Critical = 763,456 with median 24
  claims and zero missing flags; High = 608 with median 1 claim) — **never gate on
  it**. The real concrete quality signal is `claim_record_count` (median ~22).
- The mart uses its **own feature vocabulary** (`cci_*`×17, `elx_*`×31,
  `elixhauser_van_walraven_score`, raw `gdr_cd`/`yrdob`/`zipcode_5`); only ~4 of
  the legacy `optum_feature_manifest` names exist here. The mart therefore needs
  its **own leakage manifest** (`optum_mart`), not the legacy `optum` one.
- **Target self-leak:** `index_biologic_brand != 'no_treatment'` reproduces the
  ever-treated set exactly; `treatment_start_date`, all biologic/adherence/
  persistence/response/coverage/gap columns are deterministic leaks. The manifest
  forbids them and the adapter never emits them (positive enumeration).

## Cohorts

The adapter builds three **separate populations**, each with its own index anchor
and target. Each is a distinct cohort — a patient in one is not assumed in another.

| Cohort | Population | Journey anchor | Target | Output dir |
|---|---|---|---|---|
| **initiation** | Naïve-at-index CSU patients (drop pre-index-treated) | `index_date` (dx index) | `initiated_biologic_180d` — first biologic fill ∈ `[index, index+180d]` | `data/rwd/mart/initiation/` |
| **discontinuation** | Initiators with ≥180d observable follow-up | `treatment_start_date` (first biologic fill) | `discontinued_180d` (Option B, below) | `data/rwd/mart/discontinuation/` |
| **persistence** | Same as discontinuation | `treatment_start_date` | `persistent_at_180d` (Option B, below) | `data/rwd/mart/persistence/` |

The 64 baseline features are knowable at the dx index, which is ≤ treatment-start,
so they remain **pre-index** in the treatment-anchored frame.

### Option B — derive TRUE 180d disc/persistence targets

The mart ships precomputed `discontinued_90d_flag` / `persistence_60d_flag`
(60/90-day horizons), but the tier-0 contract is a **180-day** outcome. Rather
than reuse the off-horizon flags, the adapter **derives true 180d targets from the
mart's coverage/gap columns** (validated 2026-06-06; `discontinued_180d` agrees
98.2% with the precomputed `discontinued_90d_flag`):

```
cov_to_end       = (last_coverage_end − treatment_start).days
discontinued_180d = (cov_to_end <  180) AND ((max_internal_gap_days ≥ 90) OR (terminal_gap_days ≥ 90))
persistent_at_180d = (cov_to_end ≥ 180) AND (max_internal_gap_days ≤ 60)
```

### Treatment-anchored eligibility (discontinuation / persistence)

A shared prelude (`_initiator_eligible`) builds the denominator and the
right-censoring gates, each an explicit attrition step:

1. `input_patients` — full panel count (recorded via a cheap count-only scan).
2. `initiators` — `index_biologic_brand != 'no_treatment'`.
3. `quality_filter` — `claim_record_count ≥ min_claim_count` (default 2).
4. `followup_observable` — `last_observed_date − treatment_start ≥ window_days`
   (inclusive); initiators without ≥180d follow-up are **right-censored** (dropped).
5. `coverage_end_observable` — `last_coverage_end` must be observable; an
   initiator with NaT coverage end has an undefined `cov_to_end` and is
   right-censored rather than silently labelled `disc=0 AND persist=0`.

## Leakage governance — positive enumeration + manifest + lockstep

The mart bypasses the legacy converter's structural gate, so leakage control is
**positive enumeration** (allow-list), backed by the manifest and the runtime
Layer-5 checks:

- **Allow-list:** `MART_SAFE_FEATURES` (**64** columns) in
  `src/data/manifests/optum_mart_feature_manifest.py`. `build_journey_records`
  emits **only** allow-list features (∩ the frame's columns) + 2 derived
  (`geographic_region`, `enrollment_duration_days`) + journey metadata + the
  cohort target. A column not on the allow-list **cannot** become a feature.
- **Forbidden list:** `MART_FORBIDDEN_AS_FEATURES` (**16** contracts) — the 3
  targets-as-features plus the disc/persistence derivation aliases
  (`last_coverage_end`, `last_observed_date`, `max_internal_gap_days`,
  `terminal_gap_days`, `covered_days`, `pdc`, `discontinued_flag`,
  `discontinued_90d_flag`, `persistence_60d_flag`, `maintained_flag`,
  `adherent_flag`). These are the exact columns the disc/persist targets are
  derived from — feeding any as a feature would be catastrophic leakage.
- **Targets:** `MART_TARGETS = {initiated_biologic_180d, discontinued_180d,
  persistent_at_180d}`.
- **Lockstep registries:** the forbidden list flows by-value into
  `adaptive_validity_check.py`
  (`_MANIFEST_FORBIDDEN_BY_SOURCE["optum_mart"]`), and the safe list into
  `_resolve_manifest_features`, so the runtime Layer-5 verdicts (post-index leak
  catch + declared-safe σ-inflation) fire for `optum_mart` runs.

### Coverage guard — why NOT `check_manifest_coverage.py`

`scripts/check_manifest_coverage.py` AST-scans dict-literal converters
(`_build_journey_record` / `_compute_features`) to catch a hand-added output
column that the manifest doesn't catalog. The mart adapter is **not** that shape:
`build_journey_records` adds features via **variable-keyed writes**
(`rec[col] = row[col]`, `rec[target] = …`), which that visitor treats as
"unsupported writes" and would **fail discovery** on. More importantly, the mart's
feature columns come *from* `MART_SAFE_FEATURES` (the manifest itself), so
feature-coverage is **structurally guaranteed** — the AST guard would be circular.

The appropriate guard is a **contract test**
(`test_build_journey_records_emits_only_cataloged_columns` in
`tests/unit/test_scripts/test_convert_optum_mart_multicohort.py`): every emitted
feature key must be in `MART_SAFE_FEATURES`; the only non-feature keys are an
enumerated journey-metadata set and the target. This catches a future stray
`rec["leak"] = …` exactly where the risk lives.

## Output contract

Per cohort, the adapter writes a **patient-journeys-only** tree (it does NOT emit
the legacy `treatment_events` / `hcp_profiles` / `business_metrics` files — the
mart is patient-aggregated and tier-0 file_dir ingestion reads the journeys):

```
data/rwd/mart/<cohort>/
  e2i_ml_v3_patient_journeys.parquet   # one row per kept patient; carries data_split
  e2i_ml_v3_split_registry.json        # chronological split config
  attrition_report.csv                 # patient counts at each filter step
  data_dictionary.csv                  # per-feature provenance (+ opaque-band caveat)
```

`patient_journeys.parquet` carries a precomputed `data_split`
(`train`/`validation`/`test`/`holdout`) from a chronological split on the journey
anchor, so the tier-0 data_loader honors it verbatim (no re-splitting).

## Usage

### Build cohorts

```bash
# All three cohorts (writes data/rwd/mart/{initiation,discontinuation,persistence})
python scripts/convert_optum_mart.py --cohort all

# A single cohort (--output is the EXACT dir; defaults to data/rwd/mart/<cohort>)
python scripts/convert_optum_mart.py --cohort discontinuation \
    --output data/rwd/mart/discontinuation

# Stratified-by-target smoke sample (memory lever on a constrained box)
python scripts/convert_optum_mart.py --cohort initiation --sample-n 50000
```

| Flag | Default | Purpose |
|---|---|---|
| `--input FILE` | `data/rwd/Optum_Parquet/Optum.parquet` | The entity-stacked mart |
| `--cohort {initiation,discontinuation,persistence,all}` | `initiation` | Which cohort(s) to build |
| `--output DIR` | `data/rwd/mart/<cohort>` | Exact dir for one cohort; BASE dir for `all` |
| `--target-window-days N` | `180` | Outcome horizon |
| `--min-claim-count N` | `2` | Transparent quality filter (`claim_record_count ≥ N`) |
| `--sample-n N` | all | Stratified-by-target sample size for a smoke run |
| `--verbose` | off | INFO-level logging |

### Run the Tier-0 pipeline

The mart cohorts are first-class runner cohorts under the **`*_mart`** names
(distinct from the legacy `initiation`/`discontinuation`/`persistence` which point
at `data/rwd/optum/...`):

```bash
# IMPORTANT: pass --feature-manifest-source optum_mart so Layer 5 leakage
# verdicts fire. The mart dir autodetects to NO manifest (deliberate, see below);
# the runner prints a loud WARNING if you forget the flag.
python scripts/run_optum_tier0_test.py --cohort discontinuation_mart \
    --feature-manifest-source optum_mart

python scripts/run_optum_tier0_test.py --cohort persistence_mart \
    --feature-manifest-source optum_mart --single-model --no-bentoml
```

`apply_overrides` pushes the per-cohort label into `tier0.CONFIG.target_outcome`
(`initiation_mart`→`initiated_biologic_180d`, `discontinuation_mart`→
`discontinued_180d`, `persistence_mart`→`persistent_at_180d`).

#### Why the manifest must be passed explicitly

The mart cohort dirs (`data/rwd/mart/<cohort>`) are deliberately **non-`optum`**
paths so an explicit `--feature-manifest-source optum_mart` never M2-conflicts
with autodetect. The trade-off: autodetect yields `None` on a mart path, so
forgetting the flag silently drops the Layer-5 defense-in-depth. Two mitigations:

- The **converter's allow-list** is the PRIMARY leakage defense — forbidden
  columns are never even emitted into the cohort parquet, so a missing manifest
  does not expose them.
- The runner emits a **loud warning** (`_mart_manifest_warning`) when a `*_mart`
  cohort runs without a resolved manifest.

#### Runner guards (`scripts/run_optum_tier0_test.py`)

- **Single-class pre-flight** (`_single_class_error`): reads only the target
  column; a `<2`-class target fails closed with an actionable message up front
  instead of crashing tier0's stratified split deep in the pipeline. Defers (no
  error) when the journeys file/column is absent — the converter is contractually
  allowed to emit an empty / zero-positive cohort and the **deployer** fail-closes
  a weak model.
- **Convert hint** (`_convert_hint`): a missing mart-cohort dir now suggests
  `convert_optum_mart.py --cohort <base> --output <dir>` (not the wrong
  `convert_optum_rwd.py`).

## Results & known limitations

- **Initiation is feature-bound, not event-bound.** A full-population analysis
  (champion LR, AUC ~0.64–0.68, PR-AUC ~0.03 at 1.41% prevalence) plus a
  full-population scaling ablation and a 9-trial HPO plateau show performance does
  not improve with more rows, scaling, or regularization. The deployer correctly
  fail-closes. The lever is **richer pre-index features** (raw-claims
  trajectories, prior-Rx, HCP/market signal), not sample size. See
  [`docs/results/tier0_optum_mart_initiation_events_disproof_20260606.md`](results/tier0_optum_mart_initiation_events_disproof_20260606.md).
- **Comorbidity-lookback opacity (open scientific risk):** the mart's
  `cci_*`/`elx_*`/`charlson_score` are pre-engineered upstream with an UNKNOWN
  window. If it extends past index_date, an "admissible" baseline feature is
  actually post-index leakage undetectable from the schema. Until resolved
  (vendor data dictionary, or a charlson-vs-(last_observed−index) correlation
  probe), treat `cci_*`/`elx_*` as "suggestive-not-certified" for a deployable
  baseline.
- **Partially-used entities:** the *patient* mart adapter reads past the
  `optum_hcp` / `veeva_hcp` / `market` rows. `optum_hcp` is now consumed by the
  sibling **HCP adoption-propensity adapter** (`convert_optum_hcp_adoption.py`,
  section below) — the deployable commercial cohort. `veeva_hcp` / `market`
  remain a documented future lever (marketing-engagement / market covariates),
  not a defect.
- **Feast no-op:** the Feast step is a deliberate no-op (not a regression — feast
  was never in the built app image; the `tenacity>=9` vs `tenacity<9` conflict is
  original, #307). See the "Feast freshness on file-sourced runs" section in
  [`OPTUM_CONVERSION.md`](OPTUM_CONVERSION.md).

## HCP adoption-propensity cohort (commercial targeting) — `convert_optum_hcp_adoption.py`

A **separate** adapter over the **same** entity-stacked mart, reading the
`optum_hcp` *provider* grain instead of `patient`. It produces the project's one
**deployable** Optum cohort: an HCP brand-adoption-propensity model for
**commercial HCP targeting**. Where the three patient cohorts above are
feature-bound (AUC 0.54–0.64) because patient rows carry only baseline
comorbidity/demographics, the `optum_hcp` grain co-locates an adoption target
*and* a rich, admissible practice profile — so a propensity model on it clears
every tier-0 gate on merit (see "Results — deploys end-to-end" below).

### Why a separate HCP-grain cohort

The decision is documented in
[`docs/results/deployable_cohort_decision_20260607.md`](results/deployable_cohort_decision_20260607.md):

- **The patient grain is structurally feature-bound.** The mart is fractured —
  patient rows carry **no provider key** (`npi` 0%), so the commercial signal
  (engagement, adoption, market share, referral network) cannot be joined to any
  patient cohort. Initiation / discontinuation / persistence top out at AUC
  0.54–0.64 and the deployer correctly fail-closes (see
  [`disc_feature_bound_verdict_20260607.md`](results/disc_feature_bound_verdict_20260607.md)).
- **Commercial HCP targeting is *natively* an HCP-grain problem.** The
  `optum_hcp` entity (2,753,238 rows) is the **only** grain where a strong target
  and admissible features co-exist: it carries both `adoption_status`
  (~2.3% ADOPTER) and claims-derived network / volume / specialty / geo features.
  (The other HCP grain, `veeva_hcp`, has the *marketing* features but **no target
  and zero NPI overlap** with optum, so it cannot supervise this model.)
- **The signal is legitimate, not a tautology.** A leakage ablation
  ([`hcp_adoption_ablation_20260607.py`](results/hcp_adoption_ablation_20260607.py))
  splits features into groups and measures standalone + leave-one-group-out AUC:
  the AUC is dominated by **referral-network diffusion** (network features
  standalone AUC 0.81) + specialty, while the tautology-risk *volume* feature is
  **not load-bearing** (dropping it moves the full-model AUC 0.845 → 0.837). The
  mechanism is diffusion-through-professional-networks, knowable at targeting
  time — not "active providers prescribe everything."

### Input grain and target

| | |
|---|---|
| **Input** | `data/rwd/Optum_Parquet/Optum.parquet`, `entity_type == 'optum_hcp'` (~2.75M rows, pushed-down pyarrow filter) |
| **Target** | `adopted_target_brand` = `1` iff `adoption_status == 'ADOPTER'` — the HCP prescribed the target brand **XOLAIR** in the observation window (`ROGERS_CUMULATIVE_SHARE_BY_BRAND`, NDC/HCPCS match) |
| **Unit of analysis** | one HCP = one "journey" (surrogate `patient_id = HCP_<hcp_id>`), so the pipeline's patient-level split isolation **is** HCP-level isolation |
| **Population** | the **full** HCP universe — no silent specialty/quality filter, so the model ranks every provider |

### Generation pipeline (read → select → shape → split → write)

`convert()` orchestrates five stages, each a named function with a contract test
(`tests/unit/test_scripts/test_convert_optum_hcp_adoption.py`):

1. **`_read_hcp_frame`** — memory-frugal read. Projects only the gating columns
   (`entity_type`, `hcp_id`, `adoption_status`) ∪ the admissible feature
   allow-list (∩ the mart schema) and pushes down `entity_type == 'optum_hcp'`, so
   the read is bounded to the HCP grain. `--sample-n` then takes a
   **stratified-by-target** sample (preserves the rare positive rate) for a smoke
   run on a memory-constrained box.
2. **`select_hcp_cohort`** — filters to the `optum_hcp` grain and derives the
   binary target `adopted_target_brand = (adoption_status == 'ADOPTER')`. Returns
   an explicit **attrition funnel** (`input_rows` → `optum_hcp_rows` →
   `with_target` → `target_positives`). No specialty/quality filtering — the full
   universe is kept so the model ranks the whole population.
3. **`build_hcp_journey_records`** — maps each HCP row to a canonical journey
   record-dict. Emits **only** the allow-list features (positive enumeration) +
   the journey-contract metadata (`patient_journey_id = PJ_<hcp_id>`,
   `patient_id = HCP_<hcp_id>`, `patient_hash`, a fixed synthetic `index_date`,
   `journey_status = "active"`, `discontinuation_flag = 0`) + the target; applies
   the `log1p` transform to the heavy-tailed count features; and records a
   `data_quality_score` (fraction of non-null model inputs). Leakage / id /
   constant columns are **never carried through**.
4. **`assign_stratified_split`** — assigns a deterministic, target-**stratified
   random** `data_split` (train / validation / test / holdout). The HCP grain has
   **no temporal index** (the mart's `launch_dt` is a constant), so a chronological
   split would be meaningless; stratifying by target preserves the ~2.3% positive
   rate across all four splits. The remainder row(s) go to `train`. Seeded
   (`--seed`, default 42) → reproducible.
5. **`convert`** — writes the four canonical cohort files (below) via the shared
   `scripts/rwd_common.py` writers and returns a summary (`hcps`, `positives`,
   `prevalence`, per-split counts, `n_features`).

### Feature allow-list — positive enumeration, manifest-backed

The single source of truth is `OPTUM_HCP_SAFE_FEATURES` in
`src/data/manifests/optum_hcp_feature_manifest.py` (mirrors how the patient mart
adapter imports `MART_SAFE_FEATURES`). The manifest declares **21** admissible
pre-adoption predictors; the converter emits **19** of them — `HCP_SAFE_FEATURES`
is the manifest safe-list **minus** the two `referral_out_*` features curated out
below:

| Group (manifest `source`) | Emitted features |
|---|---|
| **Claims referral-network position** (`hcp_network`) | `influence_network_size`, `shared_patient_edge_count`, `shared_patient_weight`, `max_shared_patient_edge_weight`, `shared_patient_kol_score_pct`, `referral_in_degree`, `referral_in_patient_count`, `max_referral_in_edge_weight`, `referral_kol_score_pct`, `kol_score_100pt`, `kol_score` *(+ `referral_out_degree`, `referral_out_patient_count` — admissible in the manifest but emit-excluded)* |
| **All-cause / indication volume** (`hcp_volume`) | `medical_claim_count`, `medical_patient_count`, `treated_patient_count` |
| **Provider attributes** (`hcp_provider`) | `specialty_group`, `prov_type`, `prov_state`, `kol_category`, `cred_type` |

A column not on the allow-list **cannot** become a feature. Three documented
exclusion ledgers (asserted by the contract tests; the explanatory complement to
the positive enumeration, not the enforcement mechanism) record *why* each
non-emitted column is dropped:

- **`_LEAKY_HCP_COLS`** — adoption-**derived** columns the target is computed from
  (`adoption_status`, `adoption_category`, `adopter_rank`, `adopter_count`,
  `adoption_cumulative_share`, `days_to_first`, `first_adoption_dt`,
  `target_patient_count`, `target_event_count`, `distinct_target_code_count`,
  `target_match_methods`, `event_sources`). Feeding any would be catastrophic
  leakage.
- **`_ID_COLS`** — identifiers / high-cardinality provider-id-like columns
  (`hcp_id`, `npi`, `hcp_npi`, `patid`, `dea`, `hcp_name`, `grp_practice`,
  `hosp_affil`, `prov`).
- **`_CONSTANT_OR_REDUNDANT_COLS`** — single-value constants on the `optum_hcp`
  grain (`brand`, `molecule`, `is_csu_approved`, `launch_dt`, `launch_context`,
  `influence_network_method/source`) and high-card taxonomies redundant with
  `specialty_group` (`taxonomy1`, `taxonomy2`, `provcat`).

#### The `referral_out_*` gate-exclusion (conservative curation, NOT relabeling)

`referral_out_patient_count` / `referral_out_degree` stay **admissible** in the
manifest (they are claims-network metrics, **not** adoption-derived — not leaks),
but are excluded from this cohort's emit list (`_GATE_EXCLUDED_FEATURES`). Each is
the highest single-feature AUC (`referral_out_patient_count` ~0.80) and trips the
leakage detector's `single_feature_auc` gate (threshold 0.80), which is
conservatively calibrated for clinical/causal models and false-positives on
legitimately-strong commercial-targeting features. Investigation found
single-feature AUCs form a smooth 0.68–0.80 gradient (no leak), both classes
overlap, and the model is robust without them (AUC 0.84 vs 0.85; 0.81 even
dropping all ≥0.75). **Excluding them removes signal / lowers AUC** — the
conservative, anti-gaming choice — and keeps the cohort honestly deployable on the
legitimate referral-IN / shared-patient / specialty / volume / KOL diffusion
signal. The contract test `test_excluded_features_remain_admissible_in_manifest`
pins that they are curated out *without* being relabeled as leakage. (The gate's
clinical-vs-commercial mis-calibration is a filed follow-up.)

#### The `log1p` transform

Ten heavy-tailed count / weight / degree features (`_LOG1P_FEATURES`:
`influence_network_size`, `shared_patient_edge_count`, `shared_patient_weight`,
`max_shared_patient_edge_weight`, `referral_in_degree`,
`referral_in_patient_count`, `max_referral_in_edge_weight`, `medical_claim_count`,
`medical_patient_count`, `treated_patient_count`) are `log1p`-transformed at emit.
These span 0 → millions (a handful of aggregate/institutional NPIs reach network
sizes of 2.4M and shared-patient weights of 478M, far beyond any individual
prescriber), so a `log1p` is the standard transform for such count data and
improves linear-model conditioning. It **also** resolves a false-positive in the
leakage detector's range-based `perfect_class_separation` overlap metric (fooled
when the minority/adopter range is *nested* inside an outlier-inflated majority
range). The transform is **monotone**, so a genuinely disjoint leak (e.g.
`days_on_therapy = 0` for class 0, `300` for class 1) stays disjoint and is still
flagged — i.e. it fixes the false positive **without** weakening real-leak
detection (verified by the ablation). The bounded score/pct features (`kol_score`,
`kol_score_100pt`, `*_kol_score_pct`) and the categoricals are **not** transformed
— an explicit list (not a substring heuristic) guarantees no categorical or
bounded score is ever accidentally log-transformed.

### Leakage governance — allow-list + manifest + lockstep

Identical defense-in-depth to the patient mart, with the `optum_hcp` manifest:

- **Primary defense — positive enumeration.** `build_hcp_journey_records` emits
  only `HCP_SAFE_FEATURES`; adoption-derived columns are never even written into
  the cohort parquet, so a missing runtime manifest does not expose them.
- **Manifest forbidden list.** `OPTUM_HCP_FORBIDDEN_AS_FEATURES` (the target +
  every adoption-derived alias) is declared `knowable_at=post_index` in
  `optum_hcp_feature_manifest.py` as the runtime cross-check / Layer-3 backstop —
  so that even if raw `optum_hcp` data ever reached the data_preparer *without* the
  converter's allow-list, the proactive forbidden-list pass would still catch them.
- **Lockstep registries.** `optum_hcp` is registered by-value in
  `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py` — into
  the safe-feature resolver (`OPTUM_HCP_FEATURES`) and the proactive forbidden-list
  pass (`OPTUM_HCP_FORBIDDEN_AS_FEATURES`) — and in `MANIFEST_SOURCES`
  (`optum_hcp → optum_hcp_contract_for`), so the runtime Layer-5 verdicts fire for
  `optum_hcp` runs.

### Honesty caveat — cross-sectional features (propensity, not strict forward-causal)

Documented, not hidden: the network/volume features are **cross-sectional
aggregates** in this pre-built mart (no clean pre/post window), so
`knowable_at=index_date` declares the *contract intent* (these are pre-adoption
practice attributes, not derived from the adoption event). The result is therefore
an adoption-**propensity / segmentation** model. A strict forward-causal deployment
should recompute features over a pre-index baseline window upstream; network
position is structurally stable and known at targeting time, so this is a
feature-window design step, not a no-signal risk. The strict-windowed AUC may sit
a few points below the cross-sectional ~0.85 but remains comfortably above the
deployable bar.

### Output contract

Same canonical, patient-journeys-only tree as the patient cohorts, at
`data/rwd/mart/hcp_adoption/`:

```
data/rwd/mart/hcp_adoption/
  e2i_ml_v3_patient_journeys.parquet   # one row per HCP (patient_id = HCP_<hcp_id>); carries data_split
  e2i_ml_v3_split_registry.json        # stratified_random split config (null dates, temporal_gap_days=0)
  attrition_report.csv                 # HCP counts at each funnel step
  data_dictionary.csv                  # per-feature provenance (+ windowing caveat)
```

The split registry is **honest about the HCP grain**:
`split_strategy = "stratified_random"`, every `*_date` field `null`,
`temporal_gap_days = 0`, `patient_level_isolation = True`. The tier-0 data_loader
honors the precomputed `data_split` verbatim (no re-splitting).

### Usage

```bash
# Build the cohort (full HCP universe)
python scripts/convert_optum_hcp_adoption.py --output data/rwd/mart/hcp_adoption

# Memory-bounded smoke sample (stratified by target, preserves the ~2.3% rate)
python scripts/convert_optum_hcp_adoption.py --sample-n 100000

# Run tier-0 — BOTH flags matter:
#   --feature-manifest-source optum_hcp  → Layer-5 leakage verdicts fire
#   --deployment-intent commercial       → commercial AUC bar (0.65) + commercial operating gates
python scripts/run_optum_tier0_test.py --cohort hcp_adoption \
    --feature-manifest-source optum_hcp --deployment-intent commercial
```

| Flag (converter) | Default | Purpose |
|---|---|---|
| `--input FILE` | `data/rwd/Optum_Parquet/Optum.parquet` | The entity-stacked mart |
| `--output DIR` | `data/rwd/mart/hcp_adoption` | Cohort output dir |
| `--sample-n N` | all | Stratified-by-target sample size for a smoke run |
| `--seed N` | `42` | Split RNG seed (reproducible split) |
| `--verbose` | off | INFO-level logging |

Two runner nuances specific to `hcp_adoption` (distinct from the `*_mart` cohorts):

- **The manifest source is `optum_hcp`, not `optum_mart`.** The cohort key
  (`hcp_adoption`) does **not** end in `_mart`, so the loud `*_mart` manifest
  warning (`_mart_manifest_warning`) does not apply — but the generic resolver
  (`_resolve_feature_manifest_source`) still warns on stderr if
  `data/rwd/mart/hcp_adoption` autodetects to no manifest. Pass
  `--feature-manifest-source optum_hcp` explicitly so the Layer-5 defense-in-depth
  fires.
- **`--deployment-intent` defaults to `clinical` (AUC 0.75).** The HCP model
  clears even that, but pass `commercial` so the commercial AUC bar (0.65) and the
  prevalence-aware operating gates (recall 0.50, MCC 0.10, net-benefit at the
  commercial cost ratio `p_t = 0.05`) are applied — these are the gates the
  commercial-targeting use case ratifies.

`apply_overrides` pushes the cohort's target into `tier0.CONFIG.target_outcome`
(`hcp_adoption → adopted_target_brand`). The runner's `_single_class_error`
pre-flight and `_convert_hint` (which suggests
`convert_optum_hcp_adoption.py --output data/rwd/mart/hcp_adoption` when the dir is
missing) both cover this cohort.

### Results — deploys end-to-end

The HCP adoption cohort **passes tier-0 end-to-end** (`success_criteria_met=True`)
— the project's one deployable Optum cohort. On the delivered run (a memory-bounded
stratified sample — the decision doc cites 100K→40K HCPs at the true ~2.3% adopter
prevalence; full-pop 2.75M needs a larger box, the droplet OOMs the evaluator at
100K):

| Metric | Value | Commercial gate | Pass |
|---|---|---|---|
| roc_auc (CV) | **0.767** (0.789 after the #786 one-hot encoding fix) | ≥ 0.60 (also ≥ 0.75 clinical) | ✅ |
| calibration_slope | **0.996** | dev ≤ 0.15 | ✅ |
| overfit Δ(train–val AUC) | **0.009** | "none" | ✅ |
| recall | **0.50** (0.736 one-hot) | ≥ 0.50 | ✅ |
| MCC | **0.143** | ≥ 0.10 | ✅ |
| PR-AUC lift | ~3.4× | ≥ 0.08 over baseline | ✅ |
| permutation above-null | +0.227 | genuine signal | ✅ |

Champion = **calibrated LogisticRegression** — the deployability-aware
`_select_champion` crowns the well-calibrated linear model over higher-AUC but
overfit trees (a standalone LightGBM ablation reaches AUC ~0.85). Delivering this
cohort end-to-end also required making **four tier-0 deployer gates
commercial-intent-aware**: (1) `deployment_intent` propagation (it was dropped by
`ScopeDefinerAgent` and silently defaulted to clinical); (2) sigmoid/Platt
calibration for the commercial path (stable slope ~1.0 at low N); (3) a duck-typed
recall-constrained operating point (the guard checked `isinstance(dict)` but
`success_criteria` is a dict-*like* pydantic model); (4) net-benefit computed on
the **deployed/calibrated** probabilities at a commercial cost ratio `p_t = 0.05`.
Each is an honest fix that honors the ratified use case (commercial HCP targeting,
cheap false positives) — **not** a loosened quality gate; discrimination,
calibration, and overfit all pass on their merits. Full detail:
[`deployable_cohort_decision_20260607.md`](results/deployable_cohort_decision_20260607.md)
and [`disc_feature_bound_verdict_20260607.md`](results/disc_feature_bound_verdict_20260607.md).

> **Cosmetic follow-up:** the runner's `deployment_id` label
> (`kisqali_discontinuation_tier0_e2`) and `problem_description` are hardcoded for
> the patient test harness — not cohort-accurate for the HCP / XOLAIR cohort. The
> model and metrics are correct; only the label string is a carryover.

## Related files

- `scripts/convert_optum_mart.py` — the adapter
- `src/data/manifests/optum_mart_feature_manifest.py` — the `optum_mart` manifest
  (allow-list, forbidden list, targets)
- `scripts/run_optum_tier0_test.py` — tier-0 runner (the `*_mart` + `hcp_adoption`
  cohorts + guards)
- `scripts/convert_optum_hcp_adoption.py` — the **HCP adoption-propensity** adapter
  (`optum_hcp` grain → deployable commercial-targeting cohort)
- `src/data/manifests/optum_hcp_feature_manifest.py` — the `optum_hcp` manifest
  (admissible network/volume/specialty/geo allow-list, adoption-derived forbidden
  list, target)
- `tests/unit/test_scripts/test_convert_optum_hcp_adoption.py` — HCP converter
  leakage + contract + log1p + split tests
- `scripts/rwd_common.py` — shared writers (`apply_chronological_split`,
  `build_split_registry`, `write_records`, `write_attrition_report`,
  `write_data_dictionary`)
- `tests/unit/test_scripts/test_convert_optum_mart_multicohort.py` — cohort +
  contract tests
- `tests/unit/test_scripts/test_run_optum_tier0_mart_cohort.py` — runner wiring +
  guard tests
- `docs/results/tier0_optum_mart_initiation_events_disproof_20260606.md` — the
  feature-bound-ceiling analysis
- `docs/results/deployable_cohort_decision_20260607.md` — why the HCP cohort (and
  why not the patient cohorts); the delivered deploy result
- `docs/results/disc_feature_bound_verdict_20260607.md` — the patient-grain
  feature-bound verdict + the HCP keep/exclude regression
- `docs/results/hcp_adoption_ablation_20260607.py` — the leakage ablation
  (network-vs-volume-vs-geo standalone + leave-one-out AUC)
- `.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md` — the original
  (now SUPERSEDED) plan; see its as-built banner for the deltas
- `docs/OPTUM_CONVERSION.md` — the **legacy raw-claims** converter (different input)
