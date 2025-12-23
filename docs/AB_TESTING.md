# A/B Testing Guide

This guide covers the A/B testing infrastructure in E2I Causal Analytics, including experiment execution, monitoring, and analysis.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Components](#components)
- [API Reference](#api-reference)
- [Statistical Methods](#statistical-methods)
- [Best Practices](#best-practices)

---

## Overview

The A/B testing system provides end-to-end experiment execution:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Design    │ -> │ Randomize   │ -> │   Enroll    │ -> │   Monitor   │
│  (Designer) │    │  (Service)  │    │  (Service)  │    │   (Agent)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                                                         │
       v                                                         v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Pre-run   │    │   Interim   │ <- │   Analyze   │ <- │   Collect   │
│ (Twin Sim)  │    │  Analysis   │    │  (Service)  │    │   (Data)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Key Capabilities

- **Randomization**: Stratified, block, and simple randomization
- **Enrollment**: Eligibility checking, consent tracking, protocol deviation handling
- **Interim Analysis**: O'Brien-Fleming, Pocock, and Haybittle-Peto stopping rules
- **Results Analysis**: ITT, per-protocol, and heterogeneous treatment effects
- **Monitoring**: SRM detection, health checks, automated alerts
- **Fidelity Tracking**: Compare actual results with Digital Twin predictions

---

## Quick Start

### 1. Design the Experiment

Use the Experiment Designer Agent or API:

```python
# Via Experiment Designer Agent
from src.agents.experiment_designer import run_experiment_designer

result = await run_experiment_designer({
    "objective": "Test new detailing approach for Remibrutinib",
    "primary_metric": "conversion_rate",
    "target_effect_size": 0.1,
    "significance_level": 0.05,
    "power": 0.8,
})
```

### 2. Randomize Units

```python
from src.services.randomization import get_randomization_service

service = get_randomization_service()

# Get units to randomize
units = [
    {"id": "hcp_001", "territory": "NE", "specialty": "rheumatology"},
    {"id": "hcp_002", "territory": "SW", "specialty": "dermatology"},
    # ... more units
]

# Stratified randomization
assignments = await service.stratified_randomize(
    experiment_id=experiment_id,
    units=units,
    strata_columns=["territory"],
    allocation_ratio={"control": 0.5, "treatment": 0.5},
)
```

### 3. Enroll Units

```python
from src.services.enrollment import get_enrollment_service, EligibilityCriteria

service = get_enrollment_service()

# Define eligibility criteria
criteria = EligibilityCriteria(
    min_rx_history_months=6,
    min_patient_panel_size=50,
    active_in_territory=True,
    not_in_concurrent_study=True,
)

# Check and enroll each unit
for assignment in assignments:
    eligibility = await service.check_eligibility(
        experiment_id=experiment_id,
        unit=assignment["unit"],
        criteria=criteria,
    )

    if eligibility.is_eligible:
        await service.enroll_unit(
            assignment_id=assignment["id"],
            eligibility_result=eligibility,
        )
```

### 4. Run Interim Analysis

```python
from src.services.interim_analysis import get_interim_analysis_service

service = get_interim_analysis_service()

# Perform interim analysis
result = await service.perform_interim_analysis(
    experiment_id=experiment_id,
    analysis_number=1,
    control_data=control_outcomes,
    treatment_data=treatment_outcomes,
)

if result.decision == "stop_efficacy":
    print("Early stopping for efficacy - treatment is effective!")
elif result.decision == "stop_futility":
    print("Early stopping for futility - treatment unlikely to succeed")
else:
    print("Continue experiment")
```

### 5. Compute Final Results

```python
from src.services.results_analysis import get_results_analysis_service

service = get_results_analysis_service()

# Compute ITT results
results = await service.compute_itt_results(
    experiment_id=experiment_id,
    control_data=final_control_data,
    treatment_data=final_treatment_data,
)

print(f"Effect: {results.effect_estimate:.3f}")
print(f"95% CI: [{results.effect_ci_lower:.3f}, {results.effect_ci_upper:.3f}]")
print(f"P-value: {results.p_value:.4f}")
print(f"Significant: {results.is_significant}")
```

---

## Components

### RandomizationService

Handles unit-to-variant assignment.

```python
from src.services.randomization import RandomizationService, RandomizationConfig

# Configure service
config = RandomizationConfig(
    seed=42,                    # For reproducibility
    deterministic=True,         # Hash-based assignment
    default_method="stratified",
)

service = RandomizationService(config=config)

# Methods available:
# - stratified_randomize()  - Balance across strata
# - block_randomize()       - Fixed block sizes
# - multi_arm_allocate()    - 3+ treatment arms
# - verify_assignment()     - Verify unit assignment
# - get_allocation_summary() - Get allocation statistics
```

### EnrollmentService

Manages unit enrollment lifecycle.

```python
from src.services.enrollment import EnrollmentService, EnrollmentConfig

config = EnrollmentConfig(
    require_explicit_consent=True,
    max_minor_deviations=3,
    auto_exclude_on_major_deviation=True,
)

service = EnrollmentService(config=config)

# Methods available:
# - check_eligibility()          - Validate eligibility criteria
# - enroll_unit()                - Create enrollment record
# - withdraw_unit()              - Withdraw from experiment
# - mark_completed()             - Mark as completed
# - mark_excluded()              - Mark as excluded
# - record_protocol_deviation()  - Record protocol deviation
# - get_enrollment_stats()       - Get enrollment statistics
# - batch_enroll()               - Batch enrollment
```

### InterimAnalysisService

Provides interim analysis with stopping rules.

```python
from src.services.interim_analysis import InterimAnalysisService, InterimAnalysisConfig

config = InterimAnalysisConfig(
    alpha_spending_function="obrien_fleming",  # or "pocock", "haybittle_peto"
    total_alpha=0.05,
    futility_threshold=0.1,
    num_interim_analyses=3,
)

service = InterimAnalysisService(config=config)

# Methods available:
# - perform_interim_analysis()        - Full interim analysis
# - calculate_alpha_boundary()        - Get adjusted alpha threshold
# - calculate_conditional_power()     - Estimate power to detect effect
# - recommend_decision()              - Get stopping recommendation
```

### ResultsAnalysisService

Computes experiment results.

```python
from src.services.results_analysis import ResultsAnalysisService, ResultsConfig

config = ResultsConfig(
    confidence_level=0.95,
    min_sample_size=30,
    effect_type="absolute",  # or "relative"
)

service = ResultsAnalysisService(config=config)

# Methods available:
# - compute_itt_results()              - Intent-to-treat analysis
# - compute_per_protocol_results()     - Per-protocol analysis
# - compute_heterogeneous_effects()    - HTE by segment
# - check_sample_ratio_mismatch()      - SRM detection
# - compare_with_twin_prediction()     - Digital Twin fidelity
```

---

## API Reference

### Experiments Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/experiments/{id}/randomize` | POST | Randomize units |
| `/experiments/{id}/enroll` | POST | Enroll a unit |
| `/experiments/{id}/enrollments/{eid}` | DELETE | Withdraw |
| `/experiments/{id}/assignments` | GET | List assignments |
| `/experiments/{id}/enrollments` | GET | Enrollment stats |
| `/experiments/{id}/interim-analysis` | POST | Run interim analysis |
| `/experiments/{id}/results` | GET | Get results |
| `/experiments/{id}/srm-checks` | GET | SRM history |
| `/experiments/{id}/fidelity` | GET | Twin fidelity |

### Example: Randomize via API

```bash
curl -X POST http://localhost:8000/experiments/{id}/randomize \
  -H "Content-Type: application/json" \
  -d '{
    "units": [
      {"id": "hcp_001", "territory": "NE"},
      {"id": "hcp_002", "territory": "SW"}
    ],
    "method": "stratified",
    "strata_columns": ["territory"],
    "allocation_ratio": {"control": 0.5, "treatment": 0.5}
  }'
```

### Example: Get Results via API

```bash
curl http://localhost:8000/experiments/{id}/results
```

Response:

```json
{
  "experiment_id": "...",
  "analysis_type": "final",
  "control_mean": 0.142,
  "treatment_mean": 0.168,
  "effect_estimate": 0.026,
  "effect_ci_lower": 0.008,
  "effect_ci_upper": 0.044,
  "p_value": 0.0043,
  "is_significant": true,
  "sample_size_control": 1523,
  "sample_size_treatment": 1489
}
```

---

## Statistical Methods

### Alpha Spending Functions

| Function | Description | Use When |
|----------|-------------|----------|
| **O'Brien-Fleming** | Conservative early, relaxed later | Standard choice for most experiments |
| **Pocock** | Equal alpha at each interim | Need equal chance to stop at each look |
| **Haybittle-Peto** | Fixed early threshold (0.001) | Want to preserve final analysis power |

### Stopping Decisions

| Decision | Criteria |
|----------|----------|
| `continue` | p-value > boundary AND conditional power > futility threshold |
| `stop_efficacy` | p-value < boundary (strong evidence of effect) |
| `stop_futility` | conditional power < futility threshold (unlikely to succeed) |

### Sample Ratio Mismatch (SRM)

SRM detection uses chi-square test:
- **Threshold**: p < 0.01 triggers warning
- **Causes**: Randomization bugs, differential attrition, data pipeline issues

### Heterogeneous Treatment Effects

HTE analysis computes effects by segment:
- Segments: territory, specialty, decile, etc.
- Method: Stratified t-tests with Bonferroni correction

---

## Best Practices

### Design Phase

1. **Pre-screen with Digital Twin**: Run simulation before live experiment
2. **Power analysis**: Ensure adequate sample size for target effect
3. **Define stopping rules upfront**: Choose alpha spending function before starting

### Execution Phase

1. **Monitor enrollment rates**: Use health checks to catch slow enrollment
2. **Check SRM regularly**: Run every 6-12 hours
3. **Document protocol deviations**: Record all deviations for per-protocol analysis

### Analysis Phase

1. **ITT as primary**: Always report intent-to-treat as primary analysis
2. **Pre-specify segments**: Define HTE segments before unblinding
3. **Track fidelity**: Compare results with Digital Twin predictions

### Common Pitfalls

| Pitfall | Prevention |
|---------|------------|
| Peeking at results | Use formal interim analysis with alpha spending |
| SRM ignored | Automated detection with alerts |
| Post-hoc segments | Pre-register all segments in design |
| Stopping too early | Conservative boundaries (O'Brien-Fleming) |

---

## Integration with Digital Twin

The system tracks fidelity between A/B results and Digital Twin predictions:

```python
# Compare actual results with prediction
fidelity = await results_service.compare_with_twin_prediction(
    experiment_id=experiment_id,
    twin_simulation_id=simulation_id,
)

print(f"Predicted Effect: {fidelity.predicted_effect:.3f}")
print(f"Actual Effect: {fidelity.actual_effect:.3f}")
print(f"Prediction Error: {fidelity.prediction_error:.3f}")
print(f"Fidelity Score: {fidelity.fidelity_score:.2f}")
print(f"Grade: {fidelity.grade}")  # A, B, C, D, F
```

This helps calibrate the Digital Twin for future experiments.

---

## Troubleshooting

### Randomization Issues

**Problem**: Imbalanced allocation
**Solution**: Use stratified randomization with appropriate strata

**Problem**: Same unit assigned differently
**Solution**: Ensure deterministic=True and consistent seed

### Enrollment Issues

**Problem**: High ineligibility rate
**Solution**: Review criteria strictness, check data quality

**Problem**: Low consent rate
**Solution**: Consider implied consent for lower-risk experiments

### Analysis Issues

**Problem**: Wide confidence intervals
**Solution**: Collect more data or accept lower precision

**Problem**: Conflicting ITT vs PP results
**Solution**: Report both, investigate protocol adherence

---

## See Also

- [Phase 15: A/B Testing Infrastructure](mlops/phase-15-ab-testing.md)
- [Experiment Designer Agent](../src/agents/experiment_designer/README.md)
- [Digital Twin System](../src/digital_twin/README.md)
- [Experiment Lifecycle Configuration](../config/experiment_lifecycle.yaml)

---

*Last Updated: 2025-12-22*
