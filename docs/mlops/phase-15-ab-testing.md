# Phase 15: A/B Testing Infrastructure

**Goal**: Implement A/B testing execution, monitoring, and analysis capabilities

**Status**: ✅ Complete

**Dependencies**: Phase 12 (End-to-End Integration), Phase 14 (Model Monitoring)

---

## Overview

The system had experiment DESIGN capabilities (Experiment Designer Agent + Digital Twin pre-screening) but lacked EXECUTION capabilities. Phase 15 adds:
- Randomization and enrollment services
- Experiment lifecycle management
- Interim analysis with stopping rules
- Results analysis and fidelity tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         A/B Testing Infrastructure                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Existing Components           │  Phase 15 New Components               │
│  ────────────────────────────  │  ─────────────────────────────────────│
│  ✅ Experiment Designer Agent  │  ✅ RandomizationService               │
│     - Power analysis           │     - Stratified randomization         │
│     - Sample size calculation  │     - Block randomization              │
│     - Treatment config         │     - Multi-arm allocation             │
│                                │                                        │
│  ✅ Digital Twin System        │  ✅ EnrollmentService                  │
│     - Pre-screening simulation │     - Eligibility validation           │
│     - Counterfactual analysis  │     - Treatment assignment             │
│     - Uplift estimation        │     - Consent tracking                 │
│                                │                                        │
│  ✅ Database Tables            │  ✅ InterimAnalysisService             │
│     - ml_experiments           │     - O'Brien-Fleming alpha spending   │
│     - twin_simulations         │     - Pocock boundaries                │
│     - twin_fidelity_tracking   │     - Haybittle-Peto stopping rules    │
│                                │     - Conditional power calculation    │
│                                │                                        │
│  ✅ experiment_lifecycle.yaml  │  ✅ ResultsAnalysisService             │
│     - State machine config     │     - Intent-to-treat analysis         │
│     - Transition rules         │     - Per-protocol analysis            │
│     - Stopping criteria        │     - Heterogeneous treatment effects  │
│                                │     - SRM detection                    │
│                                │     - Digital Twin fidelity tracking   │
│                                │                                        │
│                                │  ✅ ExperimentMonitorAgent (Tier 3)    │
│                                │     - Health metrics monitoring        │
│                                │     - SRM detection                    │
│                                │     - Anomaly alerts                   │
│                                │                                        │
│                                │  ✅ Celery Tasks                       │
│                                │     - Scheduled interim analysis       │
│                                │     - Enrollment tracking              │
│                                │     - Results computation              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components Implemented

### Database Tables

| Table | File | Description |
|-------|------|-------------|
| `ab_experiment_assignments` | `database/ml/020_ab_testing_tables.sql` | Unit randomization assignments |
| `ab_experiment_enrollments` | `database/ml/020_ab_testing_tables.sql` | Enrollment records with consent |
| `ab_interim_analyses` | `database/ml/020_ab_testing_tables.sql` | Interim analysis history |
| `ab_experiment_results` | `database/ml/021_ab_results_tables.sql` | Final and interim results |
| `ab_srm_checks` | `database/ml/021_ab_results_tables.sql` | Sample ratio mismatch history |
| `ab_fidelity_comparisons` | `database/ml/021_ab_results_tables.sql` | Digital Twin fidelity tracking |

### Services

| Service | File | Description |
|---------|------|-------------|
| RandomizationService | `src/services/randomization.py` | Stratified/block randomization, multi-arm allocation |
| EnrollmentService | `src/services/enrollment.py` | Eligibility checking, consent tracking, protocol deviations |
| InterimAnalysisService | `src/services/interim_analysis.py` | Alpha spending, stopping decisions |
| ResultsAnalysisService | `src/services/results_analysis.py` | ITT/PP analysis, HTE, SRM detection |

### Repositories

| Repository | File | Description |
|------------|------|-------------|
| ABExperimentRepository | `src/repositories/ab_experiment.py` | Assignment and enrollment CRUD |
| ABResultsRepository | `src/repositories/ab_results.py` | Results and SRM check storage |

### Agent

| Agent | Directory | Description |
|-------|-----------|-------------|
| ExperimentMonitorAgent | `src/agents/experiment_monitor/` | LangGraph agent for experiment monitoring |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/experiments/{id}/randomize` | POST | Randomize units to variants |
| `/experiments/{id}/enroll` | POST | Enroll a unit |
| `/experiments/{id}/enrollments/{eid}` | DELETE | Withdraw enrollment |
| `/experiments/{id}/assignments` | GET | Get all assignments |
| `/experiments/{id}/enrollments` | GET | Get enrollment statistics |
| `/experiments/{id}/interim-analysis` | POST | Trigger interim analysis |
| `/experiments/{id}/results` | GET | Get experiment results |
| `/experiments/{id}/srm-checks` | GET | Get SRM check history |
| `/experiments/{id}/fidelity` | GET | Get Digital Twin fidelity |

### Celery Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| `scheduled_interim_analysis` | On-demand | Run interim analysis |
| `enrollment_health_check` | Every 12h | Check enrollment rates |
| `srm_detection_sweep` | Every 6h | Detect sample ratio mismatches |
| `compute_experiment_results` | On-demand | Compute final/interim results |
| `fidelity_tracking_update` | On-demand | Update Digital Twin fidelity |

---

## Key Features

### Randomization Methods

1. **Simple Randomization**: Hash-based deterministic assignment
2. **Stratified Randomization**: Balance across strata (e.g., territory, brand)
3. **Block Randomization**: Fixed block sizes for balance
4. **Multi-arm Allocation**: Support for >2 treatment arms

### Alpha Spending Functions

1. **O'Brien-Fleming**: Conservative early, relaxed later
2. **Pocock**: Equal alpha at each interim
3. **Haybittle-Peto**: Fixed threshold for early stopping

### Analysis Types

1. **Intent-to-Treat (ITT)**: All randomized units included
2. **Per-Protocol (PP)**: Only compliant units
3. **Heterogeneous Treatment Effects (HTE)**: By segment

### Digital Twin Integration

- Compare actual A/B results with Digital Twin predictions
- Calculate fidelity scores (0-1)
- Assign grades (A/B/C/D/F)
- Generate calibration recommendations

---

## Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| RandomizationService | 53 | ✅ Pass |
| EnrollmentService | 51 | ✅ Pass |
| InterimAnalysisService | 62 | ✅ Pass |
| ResultsAnalysisService | 65 | ✅ Pass |
| Integration Tests | 26 | ✅ Pass |
| **Total** | **231** | ✅ All Pass |

---

## Files Created

```
database/ml/
├── 020_ab_testing_tables.sql          # Execution tables
└── 021_ab_results_tables.sql          # Results tables

src/services/
├── randomization.py                   # Randomization service
├── enrollment.py                      # Enrollment service
├── interim_analysis.py                # Interim analysis service
└── results_analysis.py                # Results analysis service

src/repositories/
├── ab_experiment.py                   # Experiment repository
└── ab_results.py                      # Results repository

src/agents/experiment_monitor/
├── __init__.py
├── graph.py                           # LangGraph state machine
├── state.py                           # Agent state
└── nodes/
    ├── __init__.py
    ├── health_checker.py              # Health monitoring
    ├── srm_detector.py                # SRM detection
    ├── interim_analyzer.py            # Interim analysis
    └── alert_generator.py             # Alert generation

src/tasks/
└── ab_testing_tasks.py                # Celery tasks

src/api/routes/
└── experiments.py                     # API endpoints

tests/integration/
└── test_ab_testing_flow.py            # Integration tests

tests/unit/test_services/
├── test_randomization.py              # 53 tests
├── test_enrollment.py                 # 51 tests
├── test_interim_analysis.py           # 62 tests
└── test_results_analysis.py           # 65 tests
```

---

## Usage Examples

### Randomize Units

```python
from src.services.randomization import get_randomization_service

service = get_randomization_service()

# Stratified randomization
results = await service.stratified_randomize(
    experiment_id=experiment_id,
    units=[
        {"id": "hcp_001", "territory": "NE", "specialty": "oncology"},
        {"id": "hcp_002", "territory": "SW", "specialty": "cardiology"},
    ],
    strata_columns=["territory", "specialty"],
    allocation_ratio={"control": 0.5, "treatment": 0.5},
)
```

### Enroll Units

```python
from src.services.enrollment import get_enrollment_service

service = get_enrollment_service()

# Check eligibility
eligibility = await service.check_eligibility(
    experiment_id=experiment_id,
    unit={"id": "hcp_001", "rx_history_months": 12},
    criteria=EligibilityCriteria(min_rx_history_months=6),
)

# Enroll if eligible
if eligibility.is_eligible:
    enrollment = await service.enroll_unit(
        assignment_id=assignment_id,
        eligibility_result=eligibility,
        consent_method=ConsentMethod.DIGITAL,
    )
```

### Run Interim Analysis

```python
from src.services.interim_analysis import get_interim_analysis_service

service = get_interim_analysis_service()

# Perform interim analysis
result = await service.perform_interim_analysis(
    experiment_id=experiment_id,
    analysis_number=1,
    control_data=[1.2, 1.5, 1.8, ...],
    treatment_data=[1.6, 1.9, 2.1, ...],
)

# Check stopping decision
print(f"Decision: {result.decision}")  # 'continue', 'stop_efficacy', 'stop_futility'
print(f"Conditional Power: {result.conditional_power}")
```

### Compute Results

```python
from src.services.results_analysis import get_results_analysis_service

service = get_results_analysis_service()

# Compute ITT results
results = await service.compute_itt_results(
    experiment_id=experiment_id,
    control_data=[...],
    treatment_data=[...],
)

print(f"Effect: {results.effect_estimate} [{results.effect_ci_lower}, {results.effect_ci_upper}]")
print(f"Significant: {results.is_significant}")

# Check SRM
srm = await service.check_sample_ratio_mismatch(
    experiment_id=experiment_id,
    expected_ratio={"control": 0.5, "treatment": 0.5},
    actual_counts={"control": 4823, "treatment": 5177},
)

print(f"SRM Detected: {srm.is_srm_detected}")
```

---

## Dependencies

### Python Packages

- `scipy>=1.11.0` - Statistical tests (t-test, chi-square)
- `numpy>=1.24.0` - Numerical operations
- `pandas>=2.0.0` - Data manipulation
- `langgraph>=0.2.0` - Agent state machine

### Database

- Supabase PostgreSQL with extensions:
  - `uuid-ossp` - UUID generation
  - `pg_cron` (optional) - Scheduled jobs

---

## Success Metrics

- [x] All 4 core services implemented
- [x] All 2 repositories implemented
- [x] ExperimentMonitorAgent operational
- [x] API endpoints for full experiment lifecycle
- [x] Celery tasks for scheduled operations
- [x] 231 unit tests passing
- [x] Integration tests for end-to-end flow
- [x] Digital Twin fidelity tracking active

---

## Future Enhancements

1. **Adaptive Randomization**: Response-adaptive allocation
2. **Bayesian Stopping Rules**: Posterior probability-based decisions
3. **Multi-objective Optimization**: Pareto-optimal treatment selection
4. **Real-time Dashboards**: Grafana integration for monitoring
5. **Automated Reporting**: PDF/HTML report generation

---

## Related Documentation

- [Phase 12: End-to-End Integration](phase-12-integration.md)
- [Phase 13: Feast Feature Store](phase-13-feast-feature-store.md)
- [Phase 14: Model Monitoring](phase-14-model-monitoring.md)
- [Experiment Lifecycle Configuration](../../config/experiment_lifecycle.yaml)
- [Digital Twin System](../../src/digital_twin/README.md)

---

*Last Updated: 2025-12-22*
*Phase Status: Complete*
