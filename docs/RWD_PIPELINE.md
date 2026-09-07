# Real-World Data (RWD) Pipeline Support

**Last verified against code**: 2026-09-07. Flag tables here are re-derived from
each script's `argparse` — re-run the commands given rather than trusting the
transcription.

The Tier 0 pipeline (`scripts/run_tier0_test.py`) supports running against real-world patient journey data in addition to synthetic data. The old CSU wrapper (`run_csu_tier0_test.py`), which monkey-patched the main pipeline at runtime, was removed and its functionality consolidated into the main script via CLI flags.

**Two entry points, not one.** CSU runs through `run_tier0_test.py --data-dir`.
**Optum and mart cohorts do not** — they run through their own runner,
`scripts/run_optum_tier0_test.py --cohort <name>`, which resolves the data
directory itself and pushes the cohort's target into `tier0.CONFIG`:

```bash
python scripts/run_optum_tier0_test.py --cohort initiation
```

| `--cohort` | Target outcome | Data directory |
|------------|----------------|----------------|
| `initiation` | `initiated_biologic_180d` | `data/rwd/optum/initiation` |
| `discontinuation` | `discontinued_180d` | `data/rwd/optum/discontinuation` |
| `persistence` | `persistent_at_180d` | `data/rwd/optum/persistence` |
| `initiation_mart` | `initiated_biologic_180d` | `data/rwd/mart/initiation` |
| `discontinuation_mart` | `discontinued_180d` | `data/rwd/mart/discontinuation` |
| `persistence_mart` | `persistent_at_180d` | `data/rwd/mart/persistence` |
| `hcp_adoption` | `adopted_target_brand` | `data/rwd/mart/hcp_adoption` |

Re-derive with `grep -n 'COHORT_TARGETS\|COHORT_DIR' -A 12 scripts/run_optum_tier0_test.py`.
Passing an Optum directory to `run_tier0_test.py --data-dir` bypasses that
target wiring — see `docs/OPTUM_CONVERSION.md`.

## Usage

### Synthetic data (default)

```bash
python scripts/run_tier0_test.py
```

### Real-world data

```bash
python scripts/run_tier0_test.py \
  --data-dir data/rwd/csu \
  --brand competitor \
  --target treatment_initiated \
  --indication "Chronic Spontaneous Urticaria (CSU)"
```

### RWD-relevant CLI flags

Re-derive with `grep -n 'add_argument' -A 6 scripts/run_tier0_test.py`.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | — | Load RWD from this directory instead of generating synthetic data (e.g. `data/rwd/csu`) |
| `--brand TEXT` | — | Override `CONFIG.brand` (e.g. `competitor`) |
| `--target TEXT` | — | Override `CONFIG.target_outcome` (e.g. `treatment_initiated`) |
| `--indication TEXT` | — | Override `CONFIG.indication` |
| `--feature-manifest-source {csu,optum,synthetic,synthetic_csu}` | auto-detected from `--data-dir` | Opts the run into a cohort-specific feature manifest so Layer 5 (`adaptive_validity_check`) consults the matching `FeatureContract` registry. Auto-detection maps `data/rwd/csu` → `csu`, `data/rwd/optum` → `optum`, `data/synthetic` → `synthetic`; pass explicitly to override. Unset when neither applies, preserving the cross-cohort no-false-positive default |
| `--deployment-intent {clinical,commercial}` | `clinical` | Recalibrates the deployment AUC bar. `clinical` = literature floor 0.75; `commercial` = HCP-targeting model with a separately-cited floor and prevalence-aware operating gates. The default never silently loosens the bar — opt in explicitly |
| `--min-samples-per-split N` | `10` | Minimum viable samples per split for the `split_enforcer` gate. Lower it for small-cohort RWD (e.g. `5` for Optum n=47) |
| `--split {auto,random,combined}` | `auto` | `auto` picks a combined entity+temporal split when entity and date columns are detected, else random+stratified. `combined` forces it and errors when those columns are absent — the setting that matters most on RWD |
| `--regime NAME` | `default` | Synthetic regime. **Ignored when `--data-dir` is set** |
| `--n-total N` | scenario default | `--regime scenario_*` cohort size only; ignored for RWD |
| `--seed N` | `42` | Seed for the synthetic generator and downstream training |

All other flags (`--step`, `--dry-run`, `--no-bentoml`, `--disable-mlflow`, `--imbalanced`, `--hpo-trials`, `--no-demo-cost-matrix`, `--output-dir`, `--no-save`) work alongside these.

## Data format

The `--data-dir` directory must contain `e2i_ml_v3_patient_journeys.parquet`
**or** `e2i_ml_v3_patient_journeys.json`. `load_rwd_data` **prefers the parquet**
(the Optum converters' output) and falls back to the JSON (the CSU converter's
output); either yields the same schema, and neither present raises
`FileNotFoundError`.

Generate the CSU JSON with:

```bash
python scripts/convert_csu_rwd.py                       # uses the defaults below
python scripts/convert_csu_rwd.py \
  --input data/rwd/csu/csu_data.xlsx \
  --output data/rwd/csu
```

Both `--input` and `--output` carry defaults (`data/rwd/csu/csu_data.xlsx` and
`data/rwd/csu`), so the bare invocation is valid — it just needs the default
workbook to be present. The converter also takes `--max-patients N`,
`--lookback-days N` (mask aggregate features to `[index_date - N, index_date)`;
**unset means aggregates span the whole panel, which is post-index leakage** —
see `docs/lineage/csu_field_audit.md` §3), `--dry-run` and `--verbose`.

The Optum converters write parquet and have their own defaults:
`scripts/convert_optum_rwd.py --input data/rwd/Optum_Parquet --output data/rwd/optum --cohort {all,…}`
and `scripts/convert_optum_mart.py --input data/rwd/Optum_Parquet/Optum_enriched.parquet`
(default output `data/rwd/mart/initiation`, `--target-window-days 180`,
`--min-claim-count 2`). See `docs/OPTUM_CONVERSION.md` and
`docs/OPTUM_MART_CONVERSION.md`.

The loader (`load_rwd_data`) applies these transformations automatically:

- **Age group mapping**: Collapses granular RWD buckets (`<18`, `18-34`, `35-49`) into pipeline-expected buckets (`<50`, `50-65`, `>65`).
- **Numeric coercion**: Casts `days_on_therapy`, `hcp_visits`, `prior_treatments` to int.
- **Target column**: Ensures `treatment_initiated` is int; preserves `discontinuation_flag` as nullable numeric.
- **Journey status**: Synthesizes `journey_status` if absent.
- **Medicated-only filter**: When `--target discontinuation_flag`, automatically filters to patients with `treatment_initiated == 1` and non-null discontinuation data.

## Output files

When saving results (default behavior), the output file is prefixed based on data source:

- Synthetic: `docs/results/tier0_pipeline_run_<timestamp>.md`
- RWD: `docs/results/rwd_pipeline_run_<timestamp>.md`

RWD result files include additional metadata (data directory, target, indication) in the header.

## Migration from `run_csu_tier0_test.py`

The standalone CSU wrapper was removed on 2026-04-12. Every old invocation
translates to `run_tier0_test.py` plus the four CSU overrides shown under
[Real-world data](#real-world-data), with the wrapper's own flags (`--step`,
`--dry-run`, `--target discontinuation_flag`) passed through unchanged. No
command-by-command table is kept here — the wrapper has not existed for months.
