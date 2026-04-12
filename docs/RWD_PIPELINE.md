# Real-World Data (RWD) Pipeline Support

The Tier 0 pipeline (`scripts/run_tier0_test.py`) supports running against real-world patient journey data in addition to synthetic data. This was previously handled by a separate CSU wrapper script (`run_csu_tier0_test.py`) that monkey-patched the main pipeline at runtime. That wrapper has been removed and its functionality consolidated into the main script via CLI flags.

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

### New CLI flags

| Flag | Description | Example |
|------|-------------|---------|
| `--data-dir PATH` | Load RWD from this directory instead of generating synthetic data | `data/rwd/csu` |
| `--brand TEXT` | Override `CONFIG.brand` | `competitor` |
| `--target TEXT` | Override `CONFIG.target_outcome` | `treatment_initiated` |
| `--indication TEXT` | Override `CONFIG.indication` | `"Chronic Spontaneous Urticaria (CSU)"` |

All existing flags (`--step`, `--dry-run`, `--no-bentoml`, `--imbalanced`, etc.) continue to work alongside the new ones.

## Data format

The `--data-dir` directory must contain a file named `e2i_ml_v3_patient_journeys.json` -- a JSON array of patient records. Generate it with:

```bash
python scripts/convert_csu_rwd.py
```

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

The standalone CSU wrapper has been removed. Translate old commands as follows:

| Before | After |
|--------|-------|
| `python scripts/run_csu_tier0_test.py` | `python scripts/run_tier0_test.py --data-dir data/rwd/csu --brand competitor --target treatment_initiated --indication "Chronic Spontaneous Urticaria (CSU)"` |
| `python scripts/run_csu_tier0_test.py --target discontinuation_flag` | `python scripts/run_tier0_test.py --data-dir data/rwd/csu --brand competitor --target discontinuation_flag --indication "Chronic Spontaneous Urticaria (CSU)"` |
| `python scripts/run_csu_tier0_test.py --dry-run` | `python scripts/run_tier0_test.py --dry-run --data-dir data/rwd/csu --brand competitor --target treatment_initiated --indication "Chronic Spontaneous Urticaria (CSU)"` |
| `python scripts/run_csu_tier0_test.py --step 3` | `python scripts/run_tier0_test.py --step 3 --data-dir data/rwd/csu --brand competitor --target treatment_initiated --indication "Chronic Spontaneous Urticaria (CSU)"` |
