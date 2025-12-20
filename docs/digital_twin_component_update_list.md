# E2I Digital Twin Implementation - Component Update List

## Version: 4.2.0
## Purpose: Digital Twin Engine for A/B Test Pre-Screening

---

## Executive Summary

This implementation adds Digital Twin capabilities as a **new tool for the existing Experiment Designer agent**, following the assessment recommendation. Digital twins serve as hypothesis refinement mechanisms that pre-screen experiments before real-world deployment, NOT as replacements for actual A/B tests.

---

## 1. New Files to Create

### 1.1 Python Modules

| File | Location | Purpose |
|------|----------|---------|
| `__init__.py` | `src/digital_twin/` | Package initialization |
| `twin_generator.py` | `src/digital_twin/` | ML-based HCP/Patient twin generation |
| `simulation_engine.py` | `src/digital_twin/` | Intervention simulation execution |
| `fidelity_tracker.py` | `src/digital_twin/` | Twin model accuracy tracking |
| `twin_repository.py` | `src/digital_twin/` | Twin model persistence (MLflow integration) |
| `models/twin_models.py` | `src/digital_twin/models/` | Pydantic schemas for twins |
| `models/simulation_models.py` | `src/digital_twin/models/` | Simulation result schemas |
| `tools/simulate_intervention_tool.py` | `src/agents/experiment_designer/tools/` | LangGraph tool for Experiment Designer |
| `tools/validate_twin_fidelity_tool.py` | `src/agents/experiment_designer/tools/` | Fidelity validation tool |

### 1.2 SQL Migration

| File | Location | Purpose |
|------|----------|---------|
| `011_digital_twin_tables.sql` | `database/migrations/` | Twin models, simulations, fidelity tracking |

### 1.3 Configuration Updates

| File | Update Type | Changes |
|------|-------------|---------|
| `config/domain_vocabulary.yaml` | ADD ENUMs | twin_types, simulation_statuses, fidelity_grades |
| `config/agent_config.yaml` | ADD tools | New tools for experiment_designer agent |
| `config/digital_twin_config.yaml` | NEW | Digital twin hyperparameters |

### 1.4 Documentation

| File | Location | Purpose |
|------|----------|---------|
| `E2I_Digital_Twin_Implementation.html` | `docs/` | Interactive implementation guide |

---

## 2. Existing Files to Modify

### 2.1 Experiment Designer Agent

| File | Modification |
|------|-------------|
| `src/agents/experiment_designer/agent.py` | Add digital twin workflow integration |
| `src/agents/experiment_designer/prompts.py` | Add twin-aware prompts |
| `src/agents/experiment_designer/nodes/design_node.py` | Add twin pre-screening before design |
| `src/agents/experiment_designer/tools/__init__.py` | Export new twin tools |

### 2.2 Configuration Files

| File | Modification |
|------|-------------|
| `config/domain_vocabulary.yaml` | Add v3.2.0 entries for digital twins |
| `config/agent_config.yaml` | Add twin tools to experiment_designer |

### 2.3 API Layer

| File | Modification |
|------|-------------|
| `src/api/routes/__init__.py` | Register digital twin routes |
| `src/api/routes/digital_twin.py` | NEW: REST endpoints for twin operations |

### 2.4 Database Models

| File | Modification |
|------|-------------|
| `src/database/models/__init__.py` | Export new twin models |
| `src/database/models/twin_models.py` | NEW: SQLAlchemy models |

---

## 3. SQL Tables Summary (3 New Tables)

### 3.1 digital_twin_models
Stores trained twin generator models with MLflow integration.

```
digital_twin_models
├── model_id (PK, UUID)
├── model_name (VARCHAR)
├── twin_type (ENUM: hcp, patient, territory)
├── model_version (VARCHAR)
├── mlflow_run_id (VARCHAR) → Links to ml_experiments
├── training_config (JSONB)
├── feature_columns (TEXT[])
├── target_columns (TEXT[])
├── performance_metrics (JSONB)
├── fidelity_score (FLOAT)
├── is_active (BOOLEAN)
├── brand (VARCHAR)
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)
```

### 3.2 twin_simulations
Stores individual simulation runs with results.

```
twin_simulations
├── simulation_id (PK, UUID)
├── model_id (FK → digital_twin_models)
├── experiment_design_id (FK → ml_experiments, nullable)
├── intervention_type (VARCHAR)
├── intervention_config (JSONB)
├── twin_count (INTEGER)
├── simulated_ate (FLOAT)
├── simulated_ci_lower (FLOAT)
├── simulated_ci_upper (FLOAT)
├── simulation_status (ENUM: pending, running, completed, failed)
├── recommendation (ENUM: deploy, skip, refine)
├── recommended_sample_size (INTEGER)
├── execution_time_ms (INTEGER)
├── brand (VARCHAR)
├── created_at (TIMESTAMP)
└── completed_at (TIMESTAMP)
```

### 3.3 twin_fidelity_tracking
Tracks validation of twin predictions vs. real-world outcomes.

```
twin_fidelity_tracking
├── tracking_id (PK, UUID)
├── simulation_id (FK → twin_simulations)
├── actual_experiment_id (FK → ml_experiments, nullable)
├── simulated_ate (FLOAT)
├── actual_ate (FLOAT, nullable)
├── prediction_error (FLOAT, nullable)
├── fidelity_grade (ENUM: excellent, good, fair, poor, unvalidated)
├── validation_notes (TEXT)
├── validated_at (TIMESTAMP, nullable)
└── created_at (TIMESTAMP)
```

---

## 4. New ENUMs (domain_vocabulary.yaml v3.2.0)

```yaml
# Digital Twin ENUMs
twin_types:
  - hcp          # Healthcare Professional twins
  - patient      # Patient journey twins
  - territory    # Geographic territory twins

simulation_statuses:
  - pending      # Queued for execution
  - running      # Currently simulating
  - completed    # Successfully finished
  - failed       # Execution error

simulation_recommendations:
  - deploy       # Proceed to real A/B test
  - skip         # Do not run experiment (low predicted impact)
  - refine       # Refine intervention design and re-simulate

fidelity_grades:
  - excellent    # Prediction error < 10%
  - good         # Prediction error 10-20%
  - fair         # Prediction error 20-35%
  - poor         # Prediction error > 35%
  - unvalidated  # No real-world comparison yet
```

---

## 5. Integration Points

### 5.1 Experiment Designer Workflow Update

```
CURRENT (V4.1):
Query → Context → Design → Power → Validity → Template

UPDATED (V4.2):
Query → Context → [TWIN SIMULATION] → Design → Power → Validity → Template
                        ↓
                  IF predicted ATE < MIN_THRESHOLD:
                      RETURN skip_recommendation
                  ELSE:
                      PASS prior_estimate to Power node
```

### 5.2 MLflow Integration

- Twin models registered in MLflow Model Registry
- Training runs tracked in `ml_experiments` table
- Model artifacts stored with versioning
- Fidelity metrics logged as custom metrics

### 5.3 Memory System Integration

| Memory Type | Integration |
|-------------|-------------|
| **Episodic** | Store simulation outcomes for future design context |
| **Procedural** | Cache successful simulation → design patterns |
| **Semantic** | Twin → HCP relationships in FalkorDB |

---

## 6. Agent Tool Specifications

### 6.1 simulate_intervention Tool

```python
@tool
def simulate_intervention(
    intervention_type: str,         # e.g., "email_campaign", "call_frequency"
    intervention_config: dict,      # Intervention parameters
    target_population: str,         # "hcp", "patient", or "territory"
    brand: str,                     # Brand filter
    twin_count: int = 10000,        # Number of twins to simulate
    confidence_level: float = 0.95  # CI confidence
) -> SimulationResult:
    """
    Pre-screen an intervention using digital twins before real A/B test.
    
    Returns:
        SimulationResult with:
        - simulated_ate: Estimated Average Treatment Effect
        - confidence_interval: (lower, upper)
        - recommendation: "deploy" | "skip" | "refine"
        - recommended_sample_size: For real experiment
        - fidelity_warning: If model fidelity is degraded
    """
```

### 6.2 validate_twin_fidelity Tool

```python
@tool
def validate_twin_fidelity(
    simulation_id: str,
    actual_experiment_id: str
) -> FidelityResult:
    """
    Compare twin simulation predictions to actual experiment outcomes.
    Updates fidelity tracking and model performance metrics.
    """
```

---

## 7. Configuration Parameters

### 7.1 digital_twin_config.yaml

```yaml
digital_twin:
  # Simulation thresholds
  min_effect_threshold: 0.05        # Skip if ATE < 5%
  confidence_threshold: 0.70        # Minimum simulation confidence
  
  # Twin generation
  default_twin_count: 10000
  max_twin_count: 50000
  
  # Fidelity requirements
  min_fidelity_score: 0.7           # Warn if below
  fidelity_validation_window_days: 90
  
  # Model parameters
  model_update_frequency: weekly
  feature_importance_threshold: 0.01
  
  # Performance
  simulation_timeout_seconds: 300
  batch_size: 1000
```

---

## 8. API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/digital-twin/simulate` | Run intervention simulation |
| GET | `/api/v1/digital-twin/simulations` | List simulations |
| GET | `/api/v1/digital-twin/simulations/{id}` | Get simulation details |
| POST | `/api/v1/digital-twin/validate` | Validate simulation against actuals |
| GET | `/api/v1/digital-twin/models` | List twin models |
| GET | `/api/v1/digital-twin/models/{id}/fidelity` | Get model fidelity history |

---

## 9. Testing Requirements

### 9.1 Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/digital_twin/test_twin_generator.py` | Twin generation logic |
| `tests/digital_twin/test_simulation_engine.py` | Simulation execution |
| `tests/digital_twin/test_fidelity_tracker.py` | Fidelity calculations |

### 9.2 Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/integration/test_experiment_designer_twin.py` | Tool integration |
| `tests/integration/test_twin_mlflow.py` | MLflow integration |

### 9.3 Synthetic Benchmarks

| Test | Purpose |
|------|---------|
| `generate_synthetic_twin_data.py` | Create ground truth for validation |
| `test_twin_calibration.py` | Verify twins recover known effects |

---

## 10. Migration Checklist

### Phase 1: Infrastructure (Day 1-2)
- [ ] Create `src/digital_twin/` directory structure
- [ ] Run SQL migration `011_digital_twin_tables.sql`
- [ ] Update `domain_vocabulary.yaml` to v3.2.0
- [ ] Create `digital_twin_config.yaml`

### Phase 2: Core Implementation (Day 3-5)
- [ ] Implement `TwinGenerator` class
- [ ] Implement `SimulationEngine` class
- [ ] Implement `FidelityTracker` class
- [ ] Implement repository layer

### Phase 3: Agent Integration (Day 6-7)
- [ ] Create `simulate_intervention` tool
- [ ] Create `validate_twin_fidelity` tool
- [ ] Update Experiment Designer agent workflow
- [ ] Update agent prompts

### Phase 4: API & Testing (Day 8-10)
- [ ] Implement API endpoints
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Create synthetic benchmarks

### Phase 5: Documentation & Review (Day 11-12)
- [ ] Complete HTML documentation
- [ ] Update project structure (v4.2)
- [ ] Stakeholder review
- [ ] Production deployment planning

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Twin fidelity degradation | False recommendations | Mandatory fidelity tracking, automatic warnings |
| Simulation bias reinforcement | Missed opportunities | Always recommend real tests for promising interventions |
| Validation paradox | Cannot validate without real tests | Frame as hypothesis refinement, not replacement |
| Computational cost | Slow response times | Caching, batch processing, async simulation |

---

## 12. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Simulation accuracy | ATE error < 20% | Fidelity tracking |
| Test pre-screening rate | 30% of proposed tests skipped | Simulation recommendations |
| Design iteration reduction | 40% fewer redesigns | Experiment Designer metrics |
| Time to experiment | 50% faster | Design → deployment duration |

---

## Appendix A: Directory Structure Update

```
src/digital_twin/                    # NEW PACKAGE
├── __init__.py
├── twin_generator.py               # ML-based twin generation
├── simulation_engine.py            # Intervention simulation
├── fidelity_tracker.py             # Accuracy tracking
├── twin_repository.py              # Persistence layer
│
├── models/
│   ├── __init__.py
│   ├── twin_models.py              # Pydantic twin schemas
│   └── simulation_models.py        # Simulation result schemas
│
└── tests/
    ├── __init__.py
    ├── test_twin_generator.py
    ├── test_simulation_engine.py
    └── test_fidelity_tracker.py
```

---

*Document Version: 1.0*  
*Created: 2025-01*  
*E2I Causal Analytics Platform V4.2*
