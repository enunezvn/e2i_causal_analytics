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
> therefore a **split-and-map adapter**, not a recompute. Use the one that matches
> your input drop.

## When to use

- Your Optum drop is the single entity-stacked `Optum.parquet` mart (252 columns,
  one `entity_type` discriminator column), not the 6-file raw-claims layout.
- You need a tier-0-ready CSU biologic cohort (initiation / discontinuation /
  persistence) from that mart.

## Input — the entity-stacked mart

Single file: `data/rwd/Optum_Parquet/Optum.parquet` — **252 columns ×
3,758,007 rows**, stacked by `entity_type`:

| `entity_type` | Rows | Role |
|---|---|---|
| `patient` | 814,587 | The only rows the adapter models. One row per patient, every feature pre-aggregated. |
| `optum_hcp` | 2,753,238 | Provider rows (unused by the adapter today — see "Unused entities"). |
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
- **Unused entities:** `optum_hcp` / `veeva_hcp` / `market` rows are read past
  today. They are a documented future lever (provider/market covariates), not a
  defect.
- **Feast no-op:** the Feast step is a deliberate no-op (not a regression — feast
  was never in the built app image; the `tenacity>=9` vs `tenacity<9` conflict is
  original, #307). See the "Feast freshness on file-sourced runs" section in
  [`OPTUM_CONVERSION.md`](OPTUM_CONVERSION.md).

## Related files

- `scripts/convert_optum_mart.py` — the adapter
- `src/data/manifests/optum_mart_feature_manifest.py` — the `optum_mart` manifest
  (allow-list, forbidden list, targets)
- `scripts/run_optum_tier0_test.py` — tier-0 runner (the `*_mart` cohorts + guards)
- `scripts/rwd_common.py` — shared writers (`apply_chronological_split`,
  `build_split_registry`, `write_records`, `write_attrition_report`,
  `write_data_dictionary`)
- `tests/unit/test_scripts/test_convert_optum_mart_multicohort.py` — cohort +
  contract tests
- `tests/unit/test_scripts/test_run_optum_tier0_mart_cohort.py` — runner wiring +
  guard tests
- `docs/results/tier0_optum_mart_initiation_events_disproof_20260606.md` — the
  feature-bound-ceiling analysis
- `.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md` — the original
  (now SUPERSEDED) plan; see its as-built banner for the deltas
- `docs/OPTUM_CONVERSION.md` — the **legacy raw-claims** converter (different input)
