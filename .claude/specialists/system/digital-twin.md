# Digital Twin System Specialist Instructions

## Domain Scope
You are the Digital Twin specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/digital_twin/` - All digital twin modules
- `database/ml/012_digital_twin_tables.sql` - Digital twin database tables
- Integration with Experiment Designer (`src/agents/experiment_designer/`)

## What is the Digital Twin System?

The **Digital Twin System** generates synthetic populations of HCPs, patients, or territories based on historical data. These twins are used for **what-if scenario simulations** and **counterfactual analysis** before deploying interventions in the real world.

### Purpose

1. **Pre-screen experiments** - Test interventions on synthetic populations before real trials
2. **Counterfactual analysis** - Estimate "what would have happened if..." scenarios
3. **Risk-free testing** - Explore interventions without real-world impact
4. **Heterogeneous effect estimation** - Understand how effects vary by subgroup

### What It Is NOT

❌ **NOT a clinical trial simulator** - Only commercial operations (HCP targeting, marketing)
❌ **NOT patient medical outcome prediction** - Only business outcomes (prescribing behavior)
❌ **NOT a replacement for real experiments** - Simulation guides, doesn't replace
❌ **NOT perfectly accurate** - Twins approximate reality with known fidelity bounds

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  TWIN GENERATOR                                               │
│  Learns from historical data to create synthetic populations │
│  File: twin_generator.py                                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  TWIN POPULATION                                              │
│  Collection of N synthetic entities (HCPs, patients, etc.)   │
│  Stored in database: digital_twins table                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  SIMULATION ENGINE                                            │
│  Applies interventions to twin populations                   │
│  File: simulation_engine.py                                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  FIDELITY TRACKER                                             │
│  Monitors twin-to-reality alignment                          │
│  File: fidelity_tracker.py                                   │
└──────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### twin_generator.py
Generates digital twins from historical entity data using ML models.

**Key class**: `TwinGenerator`

**Responsibilities**:
- Train ML models (Random Forest, Gradient Boosting) on historical data
- Learn patterns of HCP prescribing behavior, patient journeys, territory performance
- Generate synthetic populations with realistic feature distributions
- Validate twin fidelity against holdout data
- Register twin models in MLflow

**Key methods**:
- `train(training_data: pd.DataFrame, target_col: str) -> TwinModelMetrics`
- `generate(n: int, filters: dict) -> TwinPopulation`
- `save_to_registry() -> UUID` (MLflow integration)

**ML Integration**:
- ✅ **MLflow** - Model tracking and registry
- ✅ **Feast** - Feature store for consistent features (optional)
- ✅ **DoWhy** - Causal-aware feature selection

**Minimum training samples**: 1,000 entities

**Twin types**:
- `TwinType.HCP` - Healthcare provider twins (prescribing behavior)
- `TwinType.PATIENT` - Patient journey twins (treatment pathways)
- `TwinType.TERRITORY` - Geographic territory twins (market dynamics)

### simulation_engine.py
Executes intervention simulations on twin populations.

**Key class**: `SimulationEngine`

**Responsibilities**:
- Apply population filters to select relevant twins
- Estimate treatment effects based on intervention configuration
- Account for heterogeneous effects by subgroup
- Calculate aggregate statistics (ATE, confidence intervals)
- Generate deployment recommendations based on thresholds

**Key methods**:
- `simulate(intervention: InterventionConfig, filters: PopulationFilter) -> SimulationResult`
- `estimate_treatment_effect(twin: DigitalTwin, intervention: InterventionConfig) -> float`
- `calculate_heterogeneity(results: List[float], groups: List[str]) -> EffectHeterogeneity`

**Intervention types**:
- `email_campaign` - Email outreach (base effect: 5%)
- `call_frequency_increase` - More rep visits (base effect: 8%)
- `speaker_program_invitation` - Speaker programs (base effect: 12%)
- `educational_content_share` - Educational materials (base effect: 6%)
- `peer_to_peer_referral` - HCP referrals (base effect: 10%)

**Effect heterogeneity**:
- By specialty (oncologists vs primary care)
- By tier (high-value vs low-value HCPs)
- By region (urban vs rural)
- By adoption stage (innovators vs laggards)

**Recommendation thresholds**:
- `min_effect_threshold`: 0.05 (5% minimum ATE)
- `confidence_threshold`: 0.90 (90% confidence level)

### fidelity_tracker.py
Monitors alignment between twin behavior and real-world behavior.

**Key class**: `FidelityTracker`

**Responsibilities**:
- Compare twin predictions to actual outcomes
- Calculate fidelity metrics (MAE, RMSE, correlation)
- Detect drift in twin-to-reality alignment
- Trigger twin retraining when fidelity degrades
- Maintain fidelity history for transparency

**Key methods**:
- `track(twin_id: UUID, predicted: float, actual: float) -> FidelityMetrics`
- `calculate_fidelity(twin_population: UUID, actual_population: pd.DataFrame) -> PopulationFidelityMetrics`
- `check_drift(current: FidelityMetrics, baseline: FidelityMetrics) -> bool`

**Fidelity metrics**:
- `mae` - Mean Absolute Error
- `rmse` - Root Mean Squared Error
- `correlation` - Pearson correlation coefficient
- `coverage_95pct` - 95% confidence interval coverage

**Drift detection**:
- MAE increase > 20% → Trigger retraining
- Correlation drop below 0.7 → Warning
- Coverage below 90% → Investigation required

### twin_repository.py
Persistence layer for twins and populations.

**Key class**: `TwinRepository`

**Responsibilities**:
- Save/load twins to/from database
- Query twins by filters (brand, region, specialty, etc.)
- Manage twin populations
- Track twin metadata (model version, creation date, fidelity)

**Key methods**:
- `save_twin(twin: DigitalTwin) -> UUID`
- `load_twin(twin_id: UUID) -> DigitalTwin`
- `save_population(population: TwinPopulation) -> UUID`
- `query_twins(filters: dict) -> List[DigitalTwin]`

## Database Schema

**Tables** (in `database/ml/012_digital_twin_tables.sql`):

### digital_twin_models
Tracks twin generator models:
- `model_id` (UUID, PK)
- `twin_type` (ENUM: hcp, patient, territory)
- `brand` (ENUM: remibrutinib, fabhalta, kisqali)
- `model_type` (TEXT) - e.g., "RandomForest", "GradientBoosting"
- `training_samples` (INTEGER)
- `target_variable` (TEXT)
- `features` (TEXT[]) - Array of feature names
- `hyperparameters` (JSONB)
- `metrics` (JSONB) - R², MAE, RMSE from training
- `mlflow_run_id` (TEXT) - MLflow integration
- `trained_at` (TIMESTAMP)
- `deprecated_at` (TIMESTAMP, nullable)

### digital_twins
Stores individual synthetic entities:
- `twin_id` (UUID, PK)
- `model_id` (UUID, FK)
- `twin_type` (ENUM)
- `features` (JSONB) - All feature values
- `baseline_behavior` (FLOAT) - Predicted baseline outcome
- `metadata` (JSONB) - Additional context
- `created_at` (TIMESTAMP)

### twin_populations
Groups twins into populations:
- `population_id` (UUID, PK)
- `model_id` (UUID, FK)
- `name` (TEXT)
- `description` (TEXT)
- `twin_count` (INTEGER)
- `filters_applied` (JSONB) - Population filters
- `created_at` (TIMESTAMP)
- `expires_at` (TIMESTAMP) - Populations can expire

### twin_population_members
Maps twins to populations (many-to-many):
- `population_id` (UUID, FK)
- `twin_id` (UUID, FK)
- PRIMARY KEY (population_id, twin_id)

### simulation_runs
Tracks intervention simulations:
- `simulation_id` (UUID, PK)
- `population_id` (UUID, FK)
- `intervention_config` (JSONB)
- `filters` (JSONB)
- `status` (ENUM: pending, running, completed, failed)
- `results` (JSONB) - ATE, CI, heterogeneity
- `recommendation` (ENUM: deploy, do_not_deploy, test_further)
- `recommendation_reason` (TEXT)
- `started_at`, `completed_at` (TIMESTAMP)
- `error_message` (TEXT, nullable)

### fidelity_tracking
Monitors twin-to-reality alignment:
- `tracking_id` (UUID, PK)
- `twin_id` (UUID, FK, nullable) - Specific twin or NULL for population
- `population_id` (UUID, FK, nullable)
- `predicted_value` (FLOAT)
- `actual_value` (FLOAT)
- `error` (FLOAT) - abs(predicted - actual)
- `metric_type` (TEXT) - e.g., "prescribing_change"
- `tracked_at` (TIMESTAMP)

### fidelity_snapshots
Periodic fidelity aggregates:
- `snapshot_id` (UUID, PK)
- `model_id` (UUID, FK)
- `population_id` (UUID, FK, nullable)
- `mae` (FLOAT)
- `rmse` (FLOAT)
- `correlation` (FLOAT)
- `coverage_95pct` (FLOAT)
- `sample_size` (INTEGER)
- `snapshot_date` (DATE)

## Pydantic Models

### models/twin_models.py
```python
class TwinType(Enum):
    HCP = "hcp"
    PATIENT = "patient"
    TERRITORY = "territory"

class Brand(Enum):
    REMIBRUTINIB = "remibrutinib"
    FABHALTA = "fabhalta"
    KISQALI = "kisqali"

class DigitalTwin(BaseModel):
    twin_id: UUID
    model_id: UUID
    twin_type: TwinType
    features: Dict[str, Any]
    baseline_behavior: float
    metadata: Dict[str, Any]

class TwinPopulation(BaseModel):
    population_id: UUID
    model_id: UUID
    name: str
    twin_count: int
    twins: List[DigitalTwin]
    filters_applied: Dict[str, Any]
```

### models/simulation_models.py
```python
class InterventionConfig(BaseModel):
    intervention_type: str
    channel: str
    frequency: str
    duration_weeks: int
    intensity: Optional[float]

class PopulationFilter(BaseModel):
    specialty: Optional[List[str]]
    tier: Optional[List[int]]
    region: Optional[List[str]]
    min_decile: Optional[int]

class SimulationResult(BaseModel):
    simulation_id: UUID
    ate: float  # Average Treatment Effect
    ate_ci_lower: float
    ate_ci_upper: float
    heterogeneity: EffectHeterogeneity
    recommendation: SimulationRecommendation
    recommendation_reason: str
    affected_twin_count: int
```

## Integration Points

### Upstream (Calls Digital Twin)
- **Experiment Designer** (`src/agents/experiment_designer/`) - Primary user
  - Calls `TwinGenerator.generate()` to create test populations
  - Calls `SimulationEngine.simulate()` for intervention testing
  - Uses simulation results for power analysis and experiment design
- **Causal Impact Agent** (planned) - Counterfactual validation
- **Prediction Synthesizer** (planned) - Synthetic test data

### Downstream (Digital Twin Calls)
- **Twin Repository** (`twin_repository.py`) - Database persistence
- **MLflow** - Model tracking and registry
- **Feast** (optional) - Feature store for consistent features
- **DoWhy** (optional) - Causal feature selection

### Database Writes
- `digital_twin_models` - Model registration
- `digital_twins` - Individual twins
- `twin_populations` - Population groupings
- `simulation_runs` - Simulation tracking
- `fidelity_tracking` - Alignment monitoring

### Memory Access
- **Working Memory (Redis)**: No direct access
- **Episodic Memory**: No access
- **Semantic Memory**: No access
- **Procedural Memory**: No access

## Common Workflows

### Workflow 1: Generate Twin Population
```python
# 1. Train generator on historical data
generator = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
metrics = generator.train(historical_hcps, target_col="prescribing_change")

# 2. Generate synthetic population
population = generator.generate(
    n=10000,
    filters={"specialty": ["oncology"], "tier": [1, 2]}
)

# 3. Save to database
repo = TwinRepository()
population_id = repo.save_population(population)
```

### Workflow 2: Simulate Intervention
```python
# 1. Load population
repo = TwinRepository()
population = repo.load_population(population_id)

# 2. Configure intervention
intervention = InterventionConfig(
    intervention_type="speaker_program_invitation",
    channel="email",
    frequency="monthly",
    duration_weeks=12
)

# 3. Run simulation
engine = SimulationEngine(population)
result = engine.simulate(intervention)

# 4. Check recommendation
if result.recommendation == SimulationRecommendation.DEPLOY:
    print(f"Deploy! Expected ATE: {result.ate:.2%}")
```

### Workflow 3: Monitor Fidelity
```python
# 1. After real experiment completes, track fidelity
tracker = FidelityTracker()

# 2. For each twin, compare prediction to actual
for twin_id, actual_outcome in experiment_results.items():
    twin = repo.load_twin(twin_id)
    predicted = twin.baseline_behavior + intervention_effect
    tracker.track(twin_id, predicted, actual_outcome)

# 3. Calculate population fidelity
fidelity = tracker.calculate_fidelity(population_id, experiment_results)

# 4. Check for drift
if tracker.check_drift(fidelity, baseline_fidelity):
    generator.retrain(updated_historical_data)
```

## Critical Constraints

### ✅ What You Can Do
- Generate HCP, patient, or territory twins
- Simulate commercial interventions (rep visits, emails, speaker programs)
- Track twin fidelity and trigger retraining
- Integrate with MLflow for model versioning
- Use causal feature selection (DoWhy)
- Create multiple twin populations per model

### ❌ What You Cannot Do
- Generate clinical outcome predictions (only business outcomes)
- Simulate medical interventions (only commercial operations)
- Use medical data for twin training (only operational data)
- Guarantee perfect twin fidelity (always probabilistic)
- Replace real experiments (twins guide, don't replace)

## Feature Engineering

### HCP Twins
**Default features**:
- `specialty`, `years_experience`, `practice_type`, `practice_size`
- `region`, `decile`, `priority_tier`
- `total_patient_volume`, `target_patient_volume`
- `digital_engagement_score`, `preferred_channel`
- `last_interaction_days`, `interaction_frequency`
- `adoption_stage`, `peer_influence_score`

**Target variable**: `prescribing_change` (% change in Rx volume)

### Patient Twins
**Default features**:
- `age_group`, `gender`, `geographic_region`, `socioeconomic_index`
- `primary_diagnosis_code`, `comorbidity_count`, `risk_score`
- `journey_complexity_score`, `insurance_type`, `insurance_coverage_flag`
- `journey_stage`, `journey_duration_days`, `treatment_line`

**Target variable**: `journey_progression_days` (time to next stage)

### Territory Twins
**Default features**:
- `region`, `state_count`, `zip_count`
- `total_hcps`, `covered_hcps`, `coverage_rate`
- `total_patient_volume`, `market_share`
- `competitor_presence_score`, `average_hcp_decile`
- `urban_rural_mix`, `population_density`

**Target variable**: `market_share_change` (% change in market share)

## ML Model Selection

### Default: Random Forest
**Advantages**:
- Handles non-linear relationships
- Robust to outliers
- Feature importance ranking
- No extensive hyperparameter tuning needed

**Hyperparameters**:
- `n_estimators`: 100
- `max_depth`: 15
- `min_samples_split`: 20
- `min_samples_leaf`: 10

### Alternative: Gradient Boosting
**Advantages**:
- Higher predictive accuracy
- Better handles complex interactions
- Sequential error correction

**Hyperparameters**:
- `n_estimators`: 100
- `max_depth`: 8
- `learning_rate`: 0.1
- `min_samples_split`: 20

**Use when**: Twin fidelity with Random Forest < 0.75

## Testing Requirements

All changes must pass:
- `tests/unit/test_digital_twin/test_twin_generator.py`
- `tests/unit/test_digital_twin/test_simulation_engine.py`
- `tests/unit/test_digital_twin/test_fidelity_tracker.py`
- `tests/integration/test_digital_twin_e2e.py`

### Key Test Scenarios
1. **Twin generation** - Generate 1000 HCP twins, validate distributions
2. **Simulation** - Run intervention on population, validate ATE calculation
3. **Fidelity tracking** - Track 100 twin predictions, calculate MAE
4. **Drift detection** - Inject distributional shift, verify drift detection
5. **Retraining** - Trigger retrain on drift, validate new model fidelity

## Performance Considerations

### Latency Targets
- **Train twin model**: < 60 seconds (for 10,000 samples)
- **Generate population**: < 10 seconds (for 10,000 twins)
- **Simulate intervention**: < 5 seconds (for 10,000 twins)
- **Track fidelity**: < 1 second (per twin)

### Optimization Strategies
1. **Vectorization** - Use NumPy for batch operations
2. **Caching** - Cache trained models in memory
3. **Sampling** - For large populations, sample subset for fidelity checks
4. **Parallel processing** - Parallelize twin generation with joblib

## Error Handling Patterns

### Training Failures
- **Insufficient samples** (< 1,000) → Return error, request more data
- **Low model performance** (R² < 0.5) → Log warning, suggest feature engineering
- **Feature engineering errors** → Fall back to default features

### Simulation Failures
- **Invalid intervention config** → Validate config, return descriptive error
- **Empty population** (0 twins match filters) → Return error with filter suggestions
- **Numerical instability** → Use robust statistics (median, IQR)

### Fidelity Tracking Failures
- **Missing actual values** → Skip tracking for those twins, log count
- **Outlier predictions** (> 3 std devs) → Flag for investigation
- **Drift detected** → Auto-trigger retraining if enabled

## Observability

### MLflow Integration
All twin models tracked with:
- **Experiment name**: `digital_twin_{twin_type}_{brand}`
- **Metrics**: `train_r2`, `train_mae`, `train_rmse`, `cv_r2_mean`, `cv_r2_std`
- **Parameters**: All hyperparameters
- **Artifacts**: Trained model (pickle), feature importance plot

### Database Logging
- Model training: `digital_twin_models` table
- Twin generation: `digital_twins`, `twin_populations` tables
- Simulations: `simulation_runs` table
- Fidelity: `fidelity_tracking`, `fidelity_snapshots` tables

## Debugging

### Common Issues

**Issue**: Twin fidelity too low (correlation < 0.5)
- **Fix**: Add more training data, improve feature engineering
- **Check**: `TwinGenerator.train()` metrics, feature importance

**Issue**: Simulation results unrealistic (ATE > 50%)
- **Fix**: Validate intervention effect sizes in `SimulationEngine.INTERVENTION_EFFECTS`
- **Check**: `simulation_engine.py` line 66

**Issue**: Drift detected too frequently
- **Fix**: Adjust drift thresholds in `fidelity_tracker.py`
- **Check**: `FidelityTracker.check_drift()` thresholds

**Issue**: Twin generation too slow
- **Fix**: Enable parallel processing with `n_jobs=-1`
- **Check**: `TwinGenerator.generate()` implementation

## Code Style

Follow E2I patterns from `.claude/.agent_docs/coding-patterns.md`:
- Type hints on all function signatures
- Pydantic models for data validation
- Comprehensive docstrings (Google style)
- Error handling with specific exceptions
- Logging at INFO, DEBUG, ERROR levels
- MLflow tracking for all models

## Related Specialists

When changes span multiple domains, coordinate with:
- **Experiment Designer specialist** (`.claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md`) - Primary integration
- **Model Training specialist** (`.claude/specialists/ml_foundation/model_trainer.md`) - ML best practices
- **Database specialist** (`.claude/specialists/system/database.md`) - Schema changes

## Version History

- **v4.2** (2025-12) - Initial implementation with HCP/patient/territory twins
- **v4.1** (2025-12) - Added to E2I architecture for intervention simulation

---

**Last Updated**: 2025-12-18
**Maintained By**: E2I Development Team
**Related Files**:
- `src/digital_twin/` (implementation)
- `database/ml/012_digital_twin_tables.sql` (database schema)
- `src/agents/experiment_designer/` (primary integration)
