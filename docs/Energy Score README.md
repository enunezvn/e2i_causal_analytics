# Energy Score Enhancement for E2I Causal Analytics

## Overview

This enhancement replaces the legacy "first success" estimator selection with a principled **energy score-based selection** strategy. Instead of using the first estimator that doesn't fail, we now evaluate all estimators and select the one with the **lowest energy score** (best causal estimate quality).

### Key Benefits

| Aspect | Legacy (First Success) | Energy Score |
|--------|----------------------|--------------|
| Selection Logic | First that works | Best quality score |
| Evaluation Scope | Single estimator | All estimators |
| Decision Basis | No failure | Causal accuracy metric |
| Explainability | Limited | Full comparison logged |
| Integration Effort | N/A | ~1 week |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Causal Impact Agent (5-Node)                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. GraphBuilder → 2. Estimation → 3. Refutation → 4. Sensitivity   │
│                         │                                            │
│                         ▼                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              EstimatorSelector (NEW)                          │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │  │
│  │  │CausalForest │ │ LinearDML  │ │  DRLearner  │ │   OLS   │ │  │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └────┬────┘ │  │
│  │         │               │               │              │      │  │
│  │         ▼               ▼               ▼              ▼      │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │            EnergyScoreCalculator                       │  │  │
│  │  │  • Treatment Balance (35%)                             │  │  │
│  │  │  • Outcome Fit (45%)                                   │  │  │
│  │  │  • Propensity Calibration (20%)                        │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  │                          │                                    │  │
│  │                          ▼                                    │  │
│  │                   Select Minimum                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                         │                                            │
│                         ▼                                            │
│               5. Interpretation (with selected estimator)            │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# From your e2i-causal-analytics root directory

# 1. Copy source files
cp -r e2i_energy_score_enhancement/src/causal_engine/* src/causal_engine/

# 2. Run migration
psql $DATABASE_URL -f e2i_energy_score_enhancement/migrations/011_energy_score_enhancement.sql

# 3. Update domain vocabulary
# Append contents of config/domain_vocabulary_v3_2_additions.yaml 
# to config/domain_vocabulary.yaml

# 4. Install any new dependencies (scipy should already be present)
pip install scipy>=1.9.0
```

## Usage

### Basic Usage

```python
from src.causal_engine.estimator_selection import select_best_estimator

# Prepare your data
treatment = df['treatment'].values
outcome = df['outcome'].values
covariates = df[['x1', 'x2', 'x3']]

# Select best estimator using energy score
result = select_best_estimator(treatment, outcome, covariates)

# Access results
print(f"Selected: {result.selected.estimator_type.value}")
print(f"ATE: {result.selected.ate:.4f}")
print(f"Energy Score: {result.selected.energy_score:.4f}")
print(f"Reason: {result.selection_reason}")

# Compare all estimators
for est_type, score in result.energy_scores.items():
    print(f"  {est_type}: {score:.4f}")
```

### With MLflow Tracking

```python
from src.causal_engine.estimator_selection import EstimatorSelector
from src.causal_engine.mlflow_integration import EnergyScoreMLflowTracker

selector = EstimatorSelector()
tracker = EnergyScoreMLflowTracker()

with tracker.start_selection_run(
    experiment_name="trigger_effectiveness",
    brand="Remibrutinib",
    region="Midwest",
    kpi_name="Trigger Precision"
):
    result = selector.select(treatment, outcome, covariates)
    tracker.log_selection_result(result)
```

### Custom Configuration

```python
from src.causal_engine.estimator_selection import (
    EstimatorSelector,
    EstimatorSelectorConfig,
    EstimatorConfig,
    EstimatorType,
    SelectionStrategy,
)
from src.causal_engine.energy_score import EnergyScoreConfig

# Custom energy score weights
energy_config = EnergyScoreConfig(
    weight_treatment_balance=0.40,
    weight_outcome_fit=0.40,
    weight_propensity_calibration=0.20,
    enable_bootstrap=True,
    n_bootstrap=200,
)

# Custom estimator chain
config = EstimatorSelectorConfig(
    strategy=SelectionStrategy.BEST_ENERGY_SCORE,
    estimators=[
        EstimatorConfig(EstimatorType.CAUSAL_FOREST, priority=1),
        EstimatorConfig(EstimatorType.DRLEARNER, priority=2),
        EstimatorConfig(EstimatorType.LINEAR_DML, priority=3),
    ],
    energy_score_config=energy_config,
    max_acceptable_energy_score=0.75,
)

selector = EstimatorSelector(config)
result = selector.select(treatment, outcome, covariates)
```

## Energy Score Components

The composite energy score is a weighted combination of three components:

### 1. Treatment Balance Score (35%)

Measures how well the IPW-adjusted covariate distributions match between treatment groups.

- **Low score (good)**: Treated and control groups are similar after adjustment
- **High score (bad)**: Significant imbalance remains

### 2. Outcome Fit Score (45%)

Measures how well the estimated treatment effects explain observed outcomes using doubly-robust residuals.

- **Low score (good)**: Estimates align well with DR pseudo-outcomes
- **High score (bad)**: Large residuals indicate poor fit

### 3. Propensity Calibration (20%)

Measures how well propensity scores match actual treatment rates across deciles.

- **Low score (good)**: Predicted propensities match observed rates
- **High score (bad)**: Miscalibrated propensity model

## Database Schema

### New Table: `estimator_evaluations`

Stores evaluation results for each estimator in the selection chain.

| Column | Type | Description |
|--------|------|-------------|
| `evaluation_id` | UUID | Primary key |
| `experiment_id` | UUID | FK to ml_experiments |
| `estimator_type` | VARCHAR(50) | Estimator identifier |
| `success` | BOOLEAN | Whether estimation succeeded |
| `ate` | DOUBLE PRECISION | Estimated ATE |
| `energy_score` | DOUBLE PRECISION | Composite energy score |
| `treatment_balance_score` | DOUBLE PRECISION | Component score |
| `outcome_fit_score` | DOUBLE PRECISION | Component score |
| `propensity_calibration` | DOUBLE PRECISION | Component score |
| `was_selected` | BOOLEAN | Whether this estimator was chosen |
| `selection_reason` | TEXT | Human-readable reason |

### Views

- `v_estimator_performance`: Aggregated performance by estimator type
- `v_energy_score_trends`: Weekly trends in energy scores
- `v_selection_comparison`: Compares energy vs legacy selection

## Monitoring

### Key Metrics to Track

1. **Selection Rate by Estimator**
   - Which estimators are being selected most often?
   - Is CausalForest always winning, or does it vary by context?

2. **Energy Score Distribution**
   - Are scores generally low (good) or high (concerning)?
   - Alert if average exceeds 0.65

3. **Energy Score Gap**
   - How much better is the selected estimator vs. alternatives?
   - Larger gaps = more confident selection

4. **Improvement vs. Legacy**
   - Query `v_selection_comparison` to measure benefit

### Sample Dashboard Query

```sql
SELECT 
    estimator_type,
    COUNT(*) as evaluations,
    ROUND(AVG(energy_score)::numeric, 3) as avg_energy,
    SUM(CASE WHEN was_selected THEN 1 ELSE 0 END) as times_selected
FROM estimator_evaluations
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY estimator_type
ORDER BY times_selected DESC;
```

## Testing

```bash
# Run all tests
pytest tests/test_energy_score.py -v

# Run with coverage
pytest tests/test_energy_score.py --cov=src/causal_engine --cov-report=html

# Run only fast tests (skip benchmarks)
pytest tests/test_energy_score.py -v -m "not slow"
```

## Migration Path

### Phase 1: Shadow Mode (Recommended First)

Run energy score selection alongside legacy, log both, but use legacy for production.

```python
config = EstimatorSelectorConfig(
    strategy=SelectionStrategy.FIRST_SUCCESS,  # Legacy behavior
)
# Energy scores still computed and logged for analysis
```

### Phase 2: Full Activation

After validating improvement via `v_selection_comparison`:

```python
config = EstimatorSelectorConfig(
    strategy=SelectionStrategy.BEST_ENERGY_SCORE,  # New behavior
)
```

## Files in This Package

```
e2i_energy_score_enhancement/
├── README.md                                   # This file
├── src/
│   └── causal_engine/
│       ├── energy_score.py                     # Core energy score implementation
│       ├── estimator_selection.py              # Enhanced selector with scoring
│       └── mlflow_integration.py               # MLflow logging integration
├── migrations/
│   └── 011_energy_score_enhancement.sql        # Database schema updates
├── config/
│   └── domain_vocabulary_v3_2_additions.yaml   # Vocabulary updates
└── tests/
    └── test_energy_score.py                    # Comprehensive tests
```

## Version Compatibility

- **E2I Version**: 4.2+
- **Python**: 3.10+
- **DoWhy**: 0.10+
- **EconML**: 0.14+
- **MLflow**: 2.0+
- **PostgreSQL**: 14+

## References

- [CausalTune Paper](https://arxiv.org/abs/xxxx) - Energy score methodology
- [Energy Statistics](https://en.wikipedia.org/wiki/Energy_distance) - Székely & Rizzo
- [DoWhy Documentation](https://www.pywhy.org/dowhy/)
- [EconML Documentation](https://econml.azurewebsites.net/)

---

**Author**: E2I Development Team  
**Version**: 4.2.0  
**Date**: December 2025
