# Digital Twin Module Tests

## Test Files Created

### 1. test_retraining_service.py (35+ tests)
Comprehensive tests for `src/digital_twin/retraining_service.py`:
- `TwinRetrainingConfig` dataclass validation
- `TwinRetrainingDecision` logic testing
- `TwinRetrainingJob` lifecycle management
- `TwinRetrainingService` main functionality:
  - Retraining need evaluation (fidelity, error, CI coverage triggers)
  - Retraining job creation and management
  - Auto-approval logic for critical degradation
  - Cooldown period enforcement
  - Training configuration generation
  - Job status tracking and completion
  - Job cancellation
- Factory function testing

**Coverage Areas:**
- Decision-making logic for retraining triggers
- Multiple trigger reasons (fidelity degradation, prediction error, CI coverage drop)
- Approval workflow (manual vs auto-approval)
- Job lifecycle (pending → training → completed/failed/cancelled)
- Configuration building based on trigger type
- Cooldown and rate-limiting logic

### 2. test_twin_repository.py (30+ tests)
Comprehensive tests for `src/digital_twin/twin_repository.py`:
- `TwinModelRepository`:
  - Model saving with MLflow and Redis integration
  - Model retrieval from cache and database
  - Listing active models with filters
  - Model deactivation
  - Fidelity score updates
- `SimulationRepository`:
  - Simulation result saving
  - Simulation retrieval and listing
  - Status updates
  - Experiment linking
- `FidelityRepository`:
  - Fidelity record creation
  - Validation updates
  - Record retrieval by simulation and model
- `TwinRepository` (facade pattern):
  - Unified API delegation tests
  - Sub-repository initialization

**Coverage Areas:**
- Database operations (Supabase integration)
- Redis caching layer
- MLflow model registry integration
- Repository pattern implementation
- Error handling for missing data

## Running the Tests

```bash
# Run all digital_twin tests
pytest tests/unit/test_digital_twin/ -v

# Run specific test file
pytest tests/unit/test_digital_twin/test_retraining_service.py -v

# Run with coverage
pytest tests/unit/test_digital_twin/ --cov=src/digital_twin --cov-report=term-missing

# Run specific test class
pytest tests/unit/test_digital_twin/test_retraining_service.py::TestTwinRetrainingService -v

# Run specific test
pytest tests/unit/test_digital_twin/test_retraining_service.py::TestTwinRetrainingService::test_evaluate_retraining_need_fidelity_degradation -v
```

## Test Coverage

### Key Source Files Covered:
- ✅ `src/digital_twin/retraining_service.py` (~579 lines) - Comprehensive coverage
- ✅ `src/digital_twin/twin_repository.py` (~883 lines) - Major paths covered

### Areas with High Test Coverage:
- Retraining decision logic (all trigger conditions)
- Job management workflows
- Repository CRUD operations
- Cache integration
- Error handling and edge cases

### Mock Dependencies:
All external dependencies are properly mocked:
- Supabase client (database operations)
- Redis client (caching)
- MLflow client (model registry)
- Celery tasks (async job execution)

## Notes

- All async functions are tested with `@pytest.mark.asyncio`
- Extensive use of fixtures for test data and mocks
- Tests follow AAA pattern (Arrange-Act-Assert)
- Edge cases and error paths are covered
- Integration between components is tested via facade pattern
