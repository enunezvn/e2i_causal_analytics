# Synthetic Data Generation & Validation

Reference documentation for the E2I Causal Analytics synthetic data system. Covers data generation with embedded causal effects, ML-compliant splitting, tiered validation, and the digital twin simulation engine.

## Table of Contents

1. [Overview](#overview)
2. [Data Generating Processes (DGPs)](#data-generating-processes-dgps)
3. [Generator Architecture](#generator-architecture)
4. [Patient Journey Generation (The Causal Core)](#patient-journey-generation-the-causal-core)
5. [ML-Compliant Data Splits](#ml-compliant-data-splits)
6. [Tiered Validation Pipeline](#tiered-validation-pipeline)
7. [Digital Twin System](#digital-twin-system)
8. [Running the Tests](#running-the-tests)
9. [Key Source Files](#key-source-files)

---

## Overview

The platform uses several complementary synthetic-data systems, built up over successive iterations. All share the same goal — data with **known causal structure** so the pipeline can be validated against ground truth — but differ in fidelity and purpose:

| System | Location | Purpose | Scale |
|--------|----------|---------|-------|
| **Synthetic v2 — clinical scenarios** | `src/ml/synthetic_v2/` | Disease-specific cohorts with full causal DAGs; tier0 `--regime` runs + RWD concurrent validation | ~6K rows/scenario (configurable) |
| **Synthetic v1 — generic-DGP generators** | `src/ml/synthetic/` | Causal validation with embedded TRUE_ATE across 5 DGPs; `synthetic-benchmarks` CI | 15K HCPs, 85K patients (3 brands) |
| **Legacy generator** | `src/ml/data_generator.py` | Quick prototyping, KPI gap-table seeding | ~200 patients, ~50 HCPs |
| **Causal-role golden sets** | `src/ml/causal_role_dgp/` + `tests/fixtures/` | Labeled feature→role examples for the `CausalRoleClassifier` | 91 literature + 74 synthetic entries |

> **Generations.** v1 (`src/ml/synthetic/`) is the original ground-truth-ATE validation system and is still exercised by the `synthetic-benchmarks` CI workflow. v2 (`src/ml/synthetic_v2/`) is the current focus — clinically-grounded, disease-specific scenarios the tier0 pipeline consumes via `--regime`. A **v3 design exists as a proposal only** (`docs/synthetic_v3_design.md`, issue #200); there is no `src/ml/synthetic_v3/` yet.

The v1 and v2 systems both embed **known causal effects** (TRUE_ATE / explicit DAGs) in the data generating process, so the causal inference pipeline can be validated against ground truth. Without this, there is no way to know if the pipeline's ATE estimates are correct or just artifacts of confounding.

### Why Synthetic Data?

1. **Ground truth validation** -- Real observational data has unknown true causal effects. Synthetic data with known TRUE_ATE lets us verify the pipeline recovers the correct answer.
2. **Reproducibility** -- Fixed random seeds (`seed=42`) produce identical datasets across runs.
3. **No PHI/PII** -- No patient data leaves the system; all data is generated.
4. **Controlled confounding** -- Each DGP introduces specific confounding structures that the pipeline must handle correctly.

---

## Synthetic v2 — Disease-Specific Clinical Scenarios

`src/ml/synthetic_v2/` is the current synthetic-data system. Where v1 generates generic confounded/heterogeneous DGPs, v2 builds **clinically-grounded cohorts for specific brand indications**, each with a full causal DAG (confounders, mediators, instruments, descendants, colliders) drawn from a literature-anchored design. These are the cohorts the tier0 pipeline consumes via `run_tier0_test.py --regime <scenario>`.

### Scenarios

| Scenario | Indication / Franchise | Outcome | Notes |
|----------|------------------------|---------|-------|
| `scenario_a` | HR+/HER2- early breast cancer (Kisqali) | 5-yr iDFS / disease progression | Diagnostic cohort |
| `scenario_a_balanced` | Same, 50:50 prevalence derivative | — | For class-balanced runs |
| `scenario_b` | IgA nephropathy (Fabhalta) | 5-yr ESKD progression | Screening cohort |
| `scenario_c` | Chronic spontaneous urticaria (remibrutinib) | 12-wk UAS7=0 response | **RWD concurrent-validation hook** against `data/rwd/csu/` |

### Architecture

| Module | Purpose |
|--------|---------|
| `scenarios/_base.py` | `ScenarioBuilder` ABC — each scenario declares `default_n_total` (~6K) and `correlation_blocks` |
| `scenarios/scenario_{a,a_balanced,b,c}.py` | Per-scenario feature sets, DAG edges, effect sizes |
| `dgp.py` | Shared primitives: per-feature i.i.d. sampling, Cholesky-injected block correlation (must stay PSD) |
| `splits.py` | ML-compliant chronological splits |
| `manifest.py` | Feature manifest (declared features, roles, post-index forbidden flags) |
| `rwd_loaders/csu_rwd.py` | Loads real CSU data for Scenario C concurrent validation |
| `yaml_loader.py`, `api.py` | `generate_scenario(scenario, seed, n_total)` entry point |

### Tier-0 `--regime` options

`run_tier0_test.py --regime <name>` accepts exactly the 7 regimes enumerated in `_VALID_REGIMES` (`scripts/run_tier0_test.py`, ~line 4374). They fall into two families: three **legacy** regimes (`default`, `adverse`, `clean`) generated by `SampleDataGenerator.ml_patients()` via `_regime_kwargs()`, and four **synthetic_v2 scenario** regimes (`scenario_*`) generated through `synthetic_v2/api.py`. The default is `default`.

| Regime | Family | Generator knobs (`_regime_kwargs`) | Cohort size | Intended outcome |
|--------|--------|-------------------------------------|-------------|------------------|
| `default` | legacy `ml_patients()` | `positive_rate=0.30`, `signal_strength=1.0`, `noise_sd=0.10`, `signalize_extra_features=False` | 1500 | Balanced baseline — the historical regime (`~13-18%` realised positive share). |
| `adverse` | legacy `ml_patients()` | `positive_rate=0.02`, `signal_strength=1.0`, `noise_sd=0.10`, `signalize_extra_features=False` | 1500 | Extreme class-imbalance to exercise the remediation / class-presence paths (`#645`/`#646`). Keeps signal config identical to `default` so the `TestAdverseRegimeE2E` contracts stay stable. |
| `clean` | legacy `ml_patients()` | `positive_rate=1.2`, `signal_strength=1.35`, `noise_sd=0.04`, `signalize_extra_features=True` | **4000** | High-signal regime designed to pass the v3 calibration + overfit gates as an honest green (`#633`/`#640`). The larger N closes the calibration-invariant `maximum_train_val_delta` overfit gate that post-hoc calibration cannot. |
| `scenario_a` | synthetic_v2 | dispatched via `synthetic_v2/api.py` (bypasses `ml_patients`) | ~6000 | HR+/HER2- early BC iDFS (Kisqali); calibrated AUC band `[0.78, 0.83]`. |
| `scenario_a_balanced` | synthetic_v2 | dispatched via `synthetic_v2/api.py` | ~6000 | `scenario_a` derivative with prevalence shifted to ~0.50 for class-balanced runs. |
| `scenario_b` | synthetic_v2 | dispatched via `synthetic_v2/api.py` | ~6000 | IgA nephropathy / ESKD screening (Fabhalta); calibrated AUC band `[0.72, 0.78]`. |
| `scenario_c` | synthetic_v2 | dispatched via `synthetic_v2/api.py` | ~6000 | Chronic spontaneous urticaria (remibrutinib); calibrated AUC band `[0.82, 0.88]`; RWD concurrent-validation hook. |

> **Note.** The `_regime_kwargs` docstring for `clean` still cites older plan values (`positive_rate=0.70`, `signal_strength=1.4`, `noise_sd=0.03`) and the `--regime` argparse help text cites `positive_rate=0.50`; the **live kwargs** at the `clean` branch are the authoritative values transcribed above (`positive_rate=1.2`, `signal_strength=1.35`, `noise_sd=0.04`). Per-regime cohort sizes come from `_regime_n_samples()` (`_DEFAULT_N_SAMPLES = 1500`; `_REGIME_N_SAMPLES = {"clean": 4000}`); `scenario_*` cohort sizes are set by their synthetic_v2 builders. Code is the source of truth.

### Leakage-free by construction

The v2 scenarios are designed so that **no feature leaks the outcome** — DAG edges are explicit and post-index features are flagged in the manifest. Because of this, the tier0 runner's `--regime scenario_*` path uses `skip_leakage_check=True`: the LLM-assisted leakage detector would otherwise false-positive on legitimate clinical features (e.g. `journey_status`). For real-world (RWD) cohorts the leakage detector and Layer-1 manifest verdicts run in full against the on-disk columns.

---

## Data Generating Processes (DGPs)

> The DGPs and generators in this and the following sections describe the **v1 generic-DGP system** (`src/ml/synthetic/`). For the current disease-specific system see [Synthetic v2](#synthetic-v2--disease-specific-clinical-scenarios) above.

Five DGP types are defined in `src/ml/synthetic/config.py`, each with a known TRUE_ATE and specific confounding structure. The pipeline must recover each TRUE_ATE within the specified tolerance (default: +/- 0.05).

### DGP Summary Table

| DGP Type | TRUE_ATE | Tolerance | Confounders | Purpose |
|----------|----------|-----------|-------------|---------|
| `simple_linear` | 0.40 | 0.05 | None | Baseline sanity check -- no confounding |
| `confounded` | 0.25 | 0.05 | `disease_severity`, `academic_hcp` | Standard confounding requiring adjustment |
| `heterogeneous` | 0.30 (avg) | 0.05 | `disease_severity`, `academic_hcp` | Segment-level CATE estimation |
| `time_series` | 0.30 | 0.05 | `disease_severity` | Lag effects with temporal decay |
| `selection_bias` | 0.35 | 0.05 | `disease_severity`, `academic_hcp` | Strong selection bias requiring IPW correction |

### Causal DAG Structure

All DGPs (except `simple_linear`) follow the same causal directed acyclic graph:

```
    disease_severity ──────┬──────────────────┐
          │                │                  │
          ▼                ▼                  ▼
    academic_hcp ──> engagement_score ──> treatment_initiated
                     (Treatment T)        (Outcome Y)
```

- **Confounders** affect both treatment assignment and outcome (creating spurious correlation)
- **Treatment** (`engagement_score`) has a TRUE causal effect on outcome
- The pipeline must adjust for confounders to recover the TRUE_ATE

### Heterogeneous DGP: CATE by Segment

The `heterogeneous` DGP has segment-specific Conditional Average Treatment Effects:

| Segment | Severity Range | CATE |
|---------|---------------|------|
| `high_severity` | > 7.0 | 0.50 |
| `medium_severity` | 4.0 - 7.0 | 0.30 |
| `low_severity` | < 4.0 | 0.15 |

The average ATE across segments is 0.30. The `heterogeneous_optimizer` agent must recover these segment-level effects.

---

## Generator Architecture

### BaseGenerator (`src/ml/synthetic/generators/base.py`)

All generators inherit from `BaseGenerator[T]`, an abstract generic class that provides:

- **Seeded RNG** -- `np.random.default_rng(seed)` for reproducible generation
- **Batched generation** -- `generate_batched()` yields DataFrames in chunks for memory efficiency
- **Result wrapping** -- `generate_with_result()` returns `GenerationResult` with timing metadata
- **Utility methods** -- `_random_choice()`, `_random_normal()`, `_random_dates()`, `_assign_splits()`, `_generate_ids()`

```python
@dataclass
class GeneratorConfig:
    seed: int = 42
    batch_size: int = 1000
    n_records: int = 1000
    brand: Optional[Brand] = None
    dgp_type: Optional[DGPType] = None
    start_date: date = date(2022, 1, 1)
    end_date: date = date(2024, 12, 31)
    verbose: bool = False
```

### Generator Inventory

11 specialized generators, each producing a single entity type:

| Generator | File | Entity Type | Dependencies |
|-----------|------|-------------|--------------|
| `HCPGenerator` | `hcp_generator.py` | HCP profiles | None (generated first) |
| `PatientGenerator` | `patient_generator.py` | Patient journeys | HCP DataFrame (optional, for FK integrity) |
| `TreatmentGenerator` | `treatment_generator.py` | Treatment events | Patient DataFrame |
| `EngagementGenerator` | `engagement_generator.py` | Engagement events | Patient + HCP DataFrames |
| `OutcomeGenerator` | `outcome_generator.py` | Clinical outcomes | Patient DataFrame |
| `PredictionGenerator` | `prediction_generator.py` | ML predictions | Patient DataFrame |
| `TriggerGenerator` | `trigger_generator.py` | Trigger alerts | HCP DataFrame |
| `BusinessMetricsGenerator` | `business_metrics_generator.py` | Business KPIs | Patient + HCP DataFrames |
| `FeatureStoreSeeder` | `feature_store_seeder.py` | Feast feature values | Patient + HCP DataFrames |
| `FeatureValueGenerator` | `feature_value_generator.py` | Raw feature values | Patient DataFrame |

### Entity Dependency Order

Generators must run in this order to maintain referential integrity:

```
1. HCPGenerator          (no dependencies)
2. PatientGenerator      (uses hcp_df for FK assignment)
3. TreatmentGenerator    (uses patient_df)
4. EngagementGenerator   (uses patient_df + hcp_df)
5. OutcomeGenerator      (uses patient_df)
6. PredictionGenerator   (uses patient_df)
7. TriggerGenerator      (uses hcp_df)
8. BusinessMetricsGenerator  (uses patient_df + hcp_df)
9. FeatureStoreSeeder    (uses patient_df + hcp_df)
```

### Default Entity Volumes (`EntityVolumes` dataclass)

| Entity | Per Brand | Total (3 brands) |
|--------|-----------|-------------------|
| HCP profiles | 5,000 | 15,000 |
| Patient journeys | 28,333 | ~85,000 |
| Treatment events/patient | 10-40 | - |
| Engagement events/patient | 2-10 | - |
| ML predictions/patient | 1-5 | - |
| Triggers/HCP | 5-20 | - |
| Outcomes/patient | 0-3 | - |

---

## Patient Journey Generation (The Causal Core)

`PatientGenerator` (`src/ml/synthetic/generators/patient_generator.py`) is the most important generator. It embeds the causal structure that the entire pipeline validates against.

### Step 1: Confounder Generation

```python
disease_severity = Normal(mean=5.0, std=2.0, clip=[0, 10])  # 0-10 scale
academic_hcp = Bernoulli(p=0.30)                             # 30% academic
```

### Step 2: Treatment Assignment (with Confounding)

Treatment (`engagement_score`) is influenced by confounders, creating the confounding that the pipeline must adjust for:

| DGP | Treatment Formula |
|-----|-------------------|
| `simple_linear` | `Uniform(0, 10)` -- no confounding |
| `confounded` / `heterogeneous` / `time_series` | `sigmoid((3.0 + 0.3*severity + 2.0*academic + noise) / 3) * 10` |
| `selection_bias` | `sigmoid((2.0 + 0.8*severity + noise) / 3) * 10` -- strong severity effect |

Higher disease severity and academic HCP affiliation both increase engagement score, creating a backdoor path from confounders through treatment to outcome.

### Step 3: Outcome Generation (with TRUE Causal Effect)

The outcome (`treatment_initiated`) is a binary variable generated via:

```
outcome_propensity = -2.0
    + TRUE_ATE * engagement_score     # <-- THE CAUSAL EFFECT
    + 0.4 * disease_severity          # Confounding path
    + 0.6 * academic_hcp              # Confounding path
    + Normal(0, 1)                    # Noise

treatment_initiated = 1 if sigmoid(outcome_propensity) > 0.5 else 0
```

For the **heterogeneous** DGP, `TRUE_ATE` is replaced by a segment-specific CATE:

```python
cate = where(severity > 7, 0.50,      # high: strong effect
       where(severity > 4, 0.30,       # medium: moderate
                           0.15))      # low: weak
```

For the **time_series** DGP, treatment is modulated by temporal decay:

```python
lag_effect = 0.85 ** arange(n)
effective_treatment = treatment * (0.5 + 0.5 * lag_effect)
```

### Step 4: Ground Truth Storage

After generation, ground truth is stored in `df.attrs`:

```python
df.attrs["true_ate"] = 0.25          # Known TRUE_ATE
df.attrs["dgp_type"] = "confounded"  # DGP used
df.attrs["confounders"] = ["disease_severity", "academic_hcp"]
```

The `GroundTruthStore` (`src/ml/synthetic/ground_truth/causal_effects.py`) provides a global registry for validating pipeline estimates against known values:

```python
effect = GroundTruthEffect(brand=Brand.KISQALI, dgp_type=DGPType.CONFOUNDED, ...)
store.store(effect)

# Later, validate pipeline output:
result = store.validate_estimate(Brand.KISQALI, DGPType.CONFOUNDED, estimated_ate=0.23)
# result["is_valid"] = True (within tolerance of 0.05)
```

### Brand-Specific HCP Specialty Distributions

The `HCPGenerator` aligns specialties with brand indications:

| Brand | Specialties |
|-------|------------|
| Remibrutinib | 50% Dermatology, 35% Allergy/Immunology, 15% Rheumatology |
| Fabhalta | 60% Hematology, 30% Internal Medicine, 10% Neurology |
| Kisqali | 100% Oncology |

---

## ML-Compliant Data Splits

Configured in `SplitBoundaries` (`src/ml/synthetic/config.py`).

### Chronological Boundaries

| Split | Date Range (legacy band) | Ratio |
|-------|-----------|-------|
| Train | 2022-01-01 to 2023-06-30 | 60% |
| Validation | 2023-07-01 to 2024-03-31 | 20% |
| Test | 2024-04-01 to 2024-09-30 | 10% (#44; was 15%) |
| Holdout | 2024-10-01 to 2024-12-31 | 10% (#44; was 5%) |

> **Note (#44, 2026-07-21)**: under the operative `--anchor-to-now` seed path,
> splits are cut on cumulative ROW share (`BaseGenerator._assign_splits`), not
> on these legacy calendar boundaries — the date columns document the historical
> non-anchored band and are consumed only by `get_split_for_date` (tests). The
> ratios are the operative policy (see `SplitBoundaries` in
> `src/ml/synthetic/config.py`).

### Anti-Leakage Guarantees

The `SplitValidator` (`src/ml/synthetic/validators/split_validator.py`) enforces:

1. **Patient-level isolation** -- No patient appears in multiple splits. Violations are `critical` severity leakage.
2. **Temporal ordering** -- Train dates must precede validation dates, which precede test dates. Overlaps are `warning` severity.
3. **7-day temporal gap** -- `temporal_gap_days=7` between splits prevents information bleeding across boundaries.
4. **Cross-dataset consistency** -- When validating multiple tables, the same patient must be in the same split across all tables.
5. **Target leakage detection** -- Checks for suspicious distribution differences between train and test targets.
6. **Ratio tolerance** -- Split ratios must be within 2% of expected values (#44; was 5% -- the looser tolerance let a legacy 60/20/15/5 frame pass borderline against the 10/10 policy).

### Validation Integration

The `SplitValidator` integrates with the `LeakageDetector` from `src/repositories/data_splitter.py` when available, adding additional checks for feature leakage and temporal consistency.

---

## Tiered Validation Pipeline

The validation system operates at two levels: **data validation** (is the generated data correct?) and **agent validation** (do agents produce correct outputs from the data?).

### Data Validation Pipeline (`src/ml/synthetic/validation/pipeline.py`)

Validates generated datasets with both **Pandera** schema validation and **Great Expectations** statistical validation:

```python
results, obs_summary = validate_pipeline_output(
    datasets={"patient_journeys": df, "hcp_profiles": hcp_df},
    dgp_type=DGPType.CONFOUNDED,
    run_pandera=True,    # Schema validation
    run_gx=True,         # Statistical expectations
    enable_observability=True,  # Log to MLflow/Opik
)
```

Each `PipelineValidationResult` combines:
- **Pandera**: Column types, nullable constraints, value ranges
- **Great Expectations**: Distribution checks, uniqueness, custom expectations
- **Observability**: Results logged to MLflow experiments and Opik spans

### Causal Validation (`src/ml/synthetic/validators/causal_validator.py`)

Validates that the causal inference pipeline can recover TRUE_ATE:

1. **ATE Estimation** -- Uses DoWhy (primary), statsmodels OLS (fallback), or simple correlation (last resort)
2. **Tolerance Check** -- `|estimated_ate - true_ate| <= tolerance`
3. **DoWhy Refutation Tests** (3 tests, must pass >= 60%):
   - **Placebo treatment** -- Randomized treatment should show ~zero effect
   - **Random common cause** -- Adding a random confounder shouldn't change the estimate
   - **Data subset** -- Using 80% of data should give a similar estimate
4. **Confounder balance** -- Standardized mean differences across treatment groups

### Agent Validation: Tier 0 Pipeline

**Tier 0** (`scripts/run_tier0_test.py`) is the full ML pipeline test. It runs 8 sequential steps:

1. **Scope Definition** -- Brand, indication, target outcome
2. **Cohort Construction** -- Generate the synthetic cohort (default regime: 1500 patients; the `clean` regime uses 4000 — see `_regime_n_samples()`)
3. **Data Preparation** -- Feature engineering, missing value handling
4. **Feature Analysis** -- Feature importance, correlation analysis
5. **Model Selection** -- Algorithm comparison (XGBoost, LightGBM, etc.)
6. **Model Training** -- HPO with 10 trials, cross-validation
7. **Model Deployment** -- Register in MLflow model registry
8. **Observability** -- Log to Opik for experiment tracking

Output is cached to `scripts/tier0_output_cache/latest.pkl` for reuse by Tier 1-5 tests.

### Agent Validation: Tier 1-5 Pipeline

**Tier 1-5** (`scripts/run_tier1_5_test.py`) tests all 13 downstream agents using cached Tier 0 output.

The Tier-0 cache (`scripts/tier0_output_cache/latest.pkl`) is now **committed** to the repo (`#600`), so a fresh-clone Tier 1-5 run no longer requires a prior ~10-min Tier-0 run. The committed fixture is regenerated deterministically by `scripts/generate_tier0_fixture.py` (`DEFAULT_ROWS = 600`, `SEED = 42`): it builds `eligible_df` from the real `SampleDataGenerator.ml_patients()` generator plus a tiny `LogisticRegression` and scalar metric summaries (sanitized — no version-fragile fitted preprocessor/encoder objects). Run a full Tier-0 pass only to refresh the fixture against fresh real metrics, or after a Tier-0 state-contract / `Tier0OutputMapper` change.

The `Tier0OutputMapper` (`src/testing/tier0_output_mapper.py`) adapts the Tier 0 state dictionary to each agent's expected input format:

```python
mapper = Tier0OutputMapper(tier0_state)

# Each method maps tier0 output to agent-specific input:
causal_input = mapper.map_to_causal_impact()      # Tier 2
gap_input = mapper.map_to_gap_analyzer()           # Tier 2
drift_input = mapper.map_to_drift_monitor()        # Tier 3
explainer_input = mapper.map_to_explainer()        # Tier 5
# ... 13 agents total
```

**tier0_data passthrough pattern**: The mapper passes `tier0_data=df` to agents. Each agent node's `_get_data()` method checks `state.get("tier0_data")` FIRST, then tries Supabase, then falls back to mock data. This ensures agents use real synthetic data in testing.

### 4-Layer Agent Output Validation

Each agent's output goes through 4 validation layers:

#### Layer 1: Contract Validation (`src/testing/contract_validator.py`)

Validates agent output against TypedDict state contracts:

- Required fields must be present (fields without `Optional` or `NotRequired`)
- Type checking for all present fields (handles `Union`, `Literal`, `Optional`, generic containers)
- Extra fields produce warnings (not errors)

```python
validator = ContractValidator()
result = validator.validate_state(output, CausalImpactState)
# result.valid, result.errors, result.type_errors, result.extra_fields
```

#### Layer 2: Quality Gates (`src/testing/agent_quality_gates.py`)

Per-agent semantic validation that checks **meaning**, not just structure:

| Agent | Semantic Check |
|-------|---------------|
| `orchestrator` | Must dispatch unique agents, response >= 50 chars |
| `causal_impact` | ATE must be numeric, must fall within its own CI |
| `gap_analyzer` | Must identify >= 1 opportunity, total_addressable_value > 0 |
| `heterogeneous_optimizer` | Must provide strategic interpretation when ATE + heterogeneity available |
| `drift_monitor` | Must cover >= 2/3 drift types (data, model, concept) |
| `experiment_designer` | Must calculate real sample sizes (not N/A) |
| `health_score` | Must provide diagnostics when component score < 0.8 |
| `prediction_synthesizer` | Single model must warn about insufficient diversity |
| `resource_optimizer` | Completed optimization must have savings/ROI > 5% |
| `explainer` | Must surface recommendations, reject meta-descriptions |
| `feedback_learner` | Completed status requires at least one learning activity |
| `tool_composer` | Reject fabricated sample sizes > 1000 (tier0 data has ~600 rows) |

#### Layer 3: Data Source Validation (`src/testing/data_source_validator.py`)

Detects whether agents used real data or silently fell back to mock data:

| Source Type | Description |
|-------------|-------------|
| `SUPABASE` | Real Supabase synthetic/production data |
| `TIER0_PASSTHROUGH` | Data passed through from tier0 pipeline |
| `COMPUTATIONAL` | Agent is purely computational (no external data) |
| `MOCK` | Mock/hardcoded fallback data |

Detection strategies:
- **health_score**: Perfect 100% scores indicate mock (real systems have variance)
- **gap_analyzer / heterogeneous_optimizer**: `MockDataConnector` in logs
- **orchestrator / tool_composer**: Always computational (routing, no data needed)

Agents with `reject_mock=True` (health_score, gap_analyzer, heterogeneous_optimizer) fail validation if mock data is detected.

#### Layer 4: Performance Thresholds

Defined in `TestConfig` in `scripts/run_tier0_test.py`:

| Metric | Threshold |
|--------|-----------|
| AUC-ROC | >= 0.55 |
| Minority class recall | >= 10% |
| Minority class precision | >= 5% |
| Min eligible patients | >= 30 |
| Refutation pass rate | >= 60% |
| ATE tolerance | +/- 0.05 |

> **Adaptive criteria (production default).** As of 2026-06-02 the `ADAPTIVE_CRITERIA` flag defaults to `"true"` (`_adaptive_criteria_enabled()` reads `os.getenv("ADAPTIVE_CRITERIA", "true")` in `src/agents/ml_foundation/scope_definer/nodes/criteria_validator.py`). With it on, a tier0 run's **model-quality** evaluation is judged against regime / N / baseline-keyed **adaptive** thresholds rather than a fixed bar. The fixed table above is therefore the **rollback** behavior: setting `ADAPTIVE_CRITERIA=false` reverts the platform to the fixed Apr-2026-baseline thresholds (`minimum_auc 0.75` / `minimum_precision 0.70` / `minimum_recall 0.65` / `minimum_f1 0.70`). For the full adaptive-criteria contract (which metrics are added/dropped per regime and N, the QC gate, and the deploy-calibrated machinery) see [docs/model_success_criteria.md](model_success_criteria.md) — that doc is the source of truth for the gate values; they are not re-derived here.

---

## Digital Twin System

The digital twin system generates synthetic populations from historical data and simulates intervention effects for A/B test pre-screening.

### TwinGenerator (`src/digital_twin/twin_generator.py`)

ML-based population synthesis. Trains on historical entity data to learn behavior patterns, then generates synthetic populations.

**Training**:
```python
generator = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
metrics = generator.train(training_data, target_col="prescribing_change")
# metrics: R2, RMSE, MAE, 5-fold CV scores, feature importances
```

- Algorithms: `RandomForestRegressor` or `GradientBoostingRegressor`
- Minimum 1,000 training samples required
- Features scaled with `StandardScaler`, categoricals encoded with `LabelEncoder`
- Feature statistics stored for generation (mean, std, min, max for numerical; distribution for categorical)

**Generation**:
```python
population = generator.generate(n=10000, seed=42)
# TwinPopulation with 10,000 DigitalTwin objects
```

Each `DigitalTwin` has:
- `features`: Dict of entity attributes (specialty, decile, engagement, etc.)
- `baseline_outcome`: Model-predicted outcome without intervention
- `baseline_propensity`: Treatment propensity score

**Twin types** with type-specific default features:
- `HCP`: specialty, years_experience, decile, digital_engagement_score, peer_influence_score, etc.
- `PATIENT`: age_group, risk_score, journey_stage, insurance_type, treatment_line, etc.
- `TERRITORY`: region, coverage_rate, market_share, growth_rate, competitor_presence, etc.

### SimulationEngine (`src/digital_twin/simulation_engine.py`)

Runs intervention simulations on twin populations.

```python
engine = SimulationEngine(population, min_effect_threshold=0.05)
result = engine.simulate(
    intervention_config=InterventionConfig(
        intervention_type="email_campaign",
        channel="email",
        duration_weeks=8,
        intensity_multiplier=1.0,
    ),
    population_filter=PopulationFilter(specialties=["oncology"]),
    confidence_level=0.95,
    calculate_heterogeneity=True,
)
```

**Simulation pipeline**:

1. **Filter population** -- Apply specialty, decile, region, adoption stage filters
2. **Calculate individual effects** -- Per-twin treatment effect based on:
   - Base effect for intervention type (e.g., email_campaign = 0.05)
   - Decile multiplier (lower decile = higher effect)
   - Engagement multiplier (more engaged = more responsive)
   - Adoption stage multiplier (laggards = more room to grow)
   - Duration factor (log-scaled diminishing returns)
   - Channel multiplier
   - Propensity weighting
   - Random noise
3. **Aggregate statistics** -- ATE, confidence interval, standard error
4. **Heterogeneity analysis** -- Effects by specialty, decile, region, adoption stage
5. **Generate recommendation**:
   - **DEPLOY**: ATE > threshold, CI excludes zero, CI width < |ATE|
   - **REFINE**: CI includes zero, or high uncertainty
   - **SKIP**: ATE below minimum threshold
6. **Calculate recommended sample size** -- For the real A/B test (power=0.80, alpha=0.05)

**Built-in intervention types**:

| Intervention | Base Effect | Key Modifier |
|-------------|-------------|--------------|
| `email_campaign` | 0.05 | Channel multiplier |
| `call_frequency_increase` | 0.08 | +0.02 per additional call |
| `speaker_program_invitation` | 0.12 | Tier multiplier (1.5x for tier 1) |
| `sample_distribution` | 0.03 | -- |
| `peer_influence_activation` | 0.10 | Requires influence score > 0.7 |
| `digital_engagement` | 0.06 | -- |

**Fidelity tracking**: If the generator model's fidelity score < 0.70, results include a warning. Simulation confidence is a weighted composite of sample size (30%), precision (30%), and model fidelity (40%).

**Caching**: `SimulationCache` (optional) stores results keyed by intervention config + population filter + model ID to avoid redundant simulations.

---

## Running the Tests

### Tier 0: Full ML Pipeline

```bash
# Full pipeline (8 steps, default regime generates 1500 patients, ~10 min)
# (the clean regime generates 4000 — see _REGIME_N_SAMPLES)
.venv/bin/python scripts/run_tier0_test.py

# Single step (e.g., step 3 only)
.venv/bin/python scripts/run_tier0_test.py --step 3

# With MLflow tracking
.venv/bin/python scripts/run_tier0_test.py --enable-mlflow

# Dry run (show what would happen)
.venv/bin/python scripts/run_tier0_test.py --dry-run

# With BentoML serving verification
.venv/bin/python scripts/run_tier0_test.py --include-bentoml
```

Output is cached to `scripts/tier0_output_cache/latest.pkl`.

### Tier 1-5: Agent Tests

```bash
# Test all 13 agents using cached tier0 output
.venv/bin/python scripts/run_tier1_5_test.py
```

Reads `scripts/tier0_output_cache/latest.pkl`, which is **committed** to the repo, so no prior Tier 0 run is needed. To regenerate the fixture deterministically (e.g. after a Tier-0 contract / mapper change):

```bash
# Rewrite the committed fixture (DEFAULT_ROWS=600, SEED=42)
.venv/bin/python scripts/generate_tier0_fixture.py

# Custom cohort size
.venv/bin/python scripts/generate_tier0_fixture.py --rows 600
```

### Unit Test Suite

```bash
# Full suite (43 batches, ~20 min)
scripts/run_tests_batched.sh

# With coverage
.venv/bin/pytest tests/ --cov --cov-report=term-missing

# Digital twin tests
.venv/bin/pytest tests/unit/test_digital_twin/ -v -n 4

# Causal engine tests
.venv/bin/pytest tests/unit/test_causal_engine/ -v -n 4
```

### Load Synthetic Data into Supabase

```bash
# Bulk-load generated data into database tables
.venv/bin/python scripts/load_synthetic_data.py
```

---

## Causal-Role Golden Sets

Separate from population data, these labeled fixtures evaluate the `CausalRoleClassifier` (the Layer-4 audit-evaluator line of work — issues #240 / #358 / #502):

| Fixture | Entries | Source |
|---------|---------|--------|
| `tests/fixtures/causal_role_golden_set.json` | 91 (3 cohorts: BC / CSU / PNH) | Literature-derived (#358) |
| `tests/fixtures/causal_role_golden_set_synthetic.json` | 74 (4 DGP scenarios) | Synthetic DGP — `src/ml/causal_role_dgp/` |

The synthetic golden set uses four hand-specified DAGs (`A1_confounder_heavy`, `A2_mediator_heavy`, `A3_descendant_collider_rich`, `A4_instrument_rich`). The compiled DSPy classifier and its AC3 verdict artifact live in `artifacts/dspy/` (`causal_role_classifier.json`, `ac3_verdict_n200.json`). Build/check scripts: `scripts/build_causal_role_golden_set.py`, `scripts/check_compile_golden_semantic_overlap.py`.

---

## Key Source Files

### Generation

| File | Purpose |
|------|---------|
| `src/ml/synthetic/config.py` | DGP configs, split boundaries, entity volumes, ground truth values |
| `src/ml/synthetic/generators/base.py` | `BaseGenerator` abstract class, `GeneratorConfig` |
| `src/ml/synthetic/generators/patient_generator.py` | Core causal data generation with embedded TRUE_ATE |
| `src/ml/synthetic/generators/hcp_generator.py` | HCP profile generation with brand-specialty alignment |
| `src/ml/synthetic/generators/treatment_generator.py` | Treatment event generation |
| `src/ml/synthetic/generators/engagement_generator.py` | Engagement event generation |
| `src/ml/synthetic/generators/outcome_generator.py` | Clinical outcome generation |
| `src/ml/synthetic/generators/prediction_generator.py` | ML prediction generation |
| `src/ml/synthetic/generators/trigger_generator.py` | Trigger alert generation |
| `src/ml/synthetic/generators/business_metrics_generator.py` | Business KPI generation |
| `src/ml/synthetic/generators/feature_store_seeder.py` | Feast feature store seeding |
| `src/ml/synthetic/generators/feature_value_generator.py` | Raw feature value generation |
| `src/ml/synthetic/ground_truth/causal_effects.py` | `GroundTruthStore` for tracking known effects |
| `src/ml/data_generator.py` | Legacy generator (200 patients, 50 HCPs, KPI gap tables) |
| `src/ml/synthetic_v2/api.py` | v2 entry point — `generate_scenario(scenario, seed, n_total)` |
| `src/ml/synthetic_v2/dgp.py` | v2 shared DGP machinery (sampling, Cholesky block correlation) |
| `src/ml/synthetic_v2/scenarios/` | v2 per-scenario builders (A, A-balanced, B, C) |
| `src/ml/causal_role_dgp/golden_set.py` | Synthetic golden-set DGP for the `CausalRoleClassifier` |
| `docs/synthetic_v3_design.md` | v3 design proposal (no implementation yet — issue #200) |

### Validation

| File | Purpose |
|------|---------|
| `src/ml/synthetic/validators/causal_validator.py` | ATE recovery validation with DoWhy refutation tests |
| `src/ml/synthetic/validators/split_validator.py` | Split integrity, leakage detection, cross-dataset consistency |
| `src/ml/synthetic/validation/pipeline.py` | Pandera + Great Expectations pipeline orchestration |
| `src/testing/tier0_output_mapper.py` | Maps Tier 0 state to agent-specific inputs (13 agents) |
| `src/testing/contract_validator.py` | TypedDict schema validation for agent outputs |
| `src/testing/agent_quality_gates.py` | Per-agent semantic quality checks (13 validators) |
| `src/testing/data_source_validator.py` | Mock data detection and data source enforcement |

### Digital Twin

| File | Purpose |
|------|---------|
| `src/digital_twin/twin_generator.py` | ML-based population synthesis (RF/GBM) |
| `src/digital_twin/simulation_engine.py` | Intervention simulation with heterogeneity analysis |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/run_tier0_test.py` | Full Tier 0 ML pipeline test (8 steps); `--regime` selects the synthetic regime |
| `scripts/run_tier1_5_test.py` | Tier 1-5 agent tests using cached Tier 0 output |
| `scripts/generate_tier0_fixture.py` | Deterministic regenerator for the committed Tier-0 cache fixture (`tier0_output_cache/latest.pkl`; `DEFAULT_ROWS=600`, `SEED=42`) |
| `scripts/load_synthetic_data.py` | Bulk-load generated data into Supabase |
| `scripts/run_tests_batched.sh` | Batched test suite runner (43 batches) |
