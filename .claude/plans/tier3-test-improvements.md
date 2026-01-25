# Implementation Plan: Tier 3 Agent Test Improvements

**Version**: 1.0
**Created**: 2026-01-24
**Status**: Approved
**Priority**: Medium

## Executive Summary

This plan addresses three test improvement areas identified during Tier 3 agent evaluation:

| Area | Current | Target | Priority |
|------|---------|--------|----------|
| Concept Drift Edge Cases | 6 tests | ~27 tests | P1 |
| Twin Simulation Tool Tests | 0 tests | ~40 tests | P2 |
| Performance Regression Tests | 0 dedicated | ~23 tests | P3 |

**Total New Tests**: ~84

---

## 1. Concept Drift Edge Case Tests (Priority 1)

### Current State
- **File**: `tests/unit/test_agents/test_drift_monitor/test_concept_drift.py`
- **Current Tests**: 6 tests (basic execution only)
- **Gap**: Data drift has 24+ tests; concept drift lacks edge cases

### Implementation

**File to Modify**: `tests/unit/test_agents/test_drift_monitor/test_concept_drift.py`

#### New Test Classes

```python
class TestPerformanceDegradation:
    """Test _detect_performance_degradation method."""

    def test_detects_critical_accuracy_drop()    # >20% drop
    def test_detects_high_accuracy_drop()        # >10% drop
    def test_detects_medium_accuracy_drop()      # >5% drop
    def test_no_detection_stable_accuracy()      # No significant drop
    def test_handles_empty_actual_labels()       # Edge case
    def test_handles_minimum_samples_threshold() # Below 50 samples


class TestCorrelationDrift:
    """Test _detect_correlation_drift method."""

    async def test_detects_large_correlation_change()     # >0.5 change
    async def test_detects_moderate_correlation_change()  # >0.3 change
    async def test_no_detection_stable_correlation()
    async def test_handles_missing_features()
    async def test_handles_mismatched_lengths()


class TestFisherZTest:
    """Test _fisher_z_test method."""

    def test_identical_correlations()
    def test_different_correlations()
    def test_edge_correlation_values()  # -0.999, 0.999
    def test_small_sample_sizes()


class TestSeverityDetermination:
    """Test _determine_*_severity methods."""

    def test_performance_severity_critical()
    def test_performance_severity_high()
    def test_performance_severity_medium()
    def test_performance_severity_low()
    def test_correlation_severity_levels()
    def test_significance_boundary()


class TestEdgeCases:
    """Test edge cases matching data drift coverage."""

    async def test_empty_dataset()              # No predictions
    async def test_single_feature_scenario()    # One feature only
    async def test_high_dimensional_features()  # 100+ features
    async def test_gradual_drift_pattern()      # Slow change
    async def test_sudden_drift_pattern()       # Abrupt change
    async def test_recurring_drift_pattern()    # Seasonal patterns
```

**Estimated Tests**: +21 new tests

### Dependencies
- None - uses existing `numpy`, `scipy.stats` imports
- Follow patterns from `test_data_drift.py`

---

## 2. Twin Simulation Tool Unit Tests (Priority 2)

### Current State
- **Tools**:
  - `src/agents/experiment_designer/tools/simulate_intervention_tool.py`
  - `src/agents/experiment_designer/tools/validate_twin_fidelity_tool.py`
- **Current Tests**: Integration tests only (no dedicated unit tests)

### Implementation

#### File 1: `tests/unit/test_agents/test_experiment_designer/test_simulate_intervention_tool.py`

```python
"""Tests for Simulate Intervention Tool."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionInput:
    """Test input schema validation."""

    def test_valid_input_minimal()
    def test_valid_input_full()
    def test_invalid_brand()
    def test_invalid_target_population()
    def test_invalid_twin_count_range()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateIntervention:
    """Test simulate_intervention tool function."""

    def test_returns_deploy_recommendation()
    def test_returns_skip_recommendation()
    def test_returns_refine_recommendation()
    def test_confidence_interval_calculated()
    def test_top_segments_returned()
    def test_recommended_sample_size_calculated()
    def test_handles_simulation_error()
    def test_fidelity_warning_included()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestDigitalTwinWorkflow:
    """Test DigitalTwinWorkflow class."""

    def test_propose_experiment_skip()
    def test_propose_experiment_design()
    def test_passes_prior_estimate()
    def test_includes_top_segments()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestMockResultGeneration:
    """Test _create_mock_result function."""

    def test_mock_result_structure()
    def test_mock_result_intervention_types()
    def test_mock_result_confidence_bounds()
```

**Estimated Tests**: 20 tests

#### File 2: `tests/unit/test_agents/test_experiment_designer/test_validate_twin_fidelity_tool.py`

```python
"""Tests for Validate Twin Fidelity Tool."""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestValidateFidelityInput:
    """Test input schema validation."""

    def test_valid_input_minimal()
    def test_valid_input_full()
    def test_invalid_simulation_id_format()
    def test_confounding_factors_list()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestValidateTwinFidelity:
    """Test validate_twin_fidelity tool function."""

    def test_excellent_prediction()
    def test_good_prediction()
    def test_fair_prediction()
    def test_poor_prediction()
    def test_prediction_error_calculation()
    def test_ci_coverage_check()
    def test_model_update_recommended()
    def test_handles_missing_simulation()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestGetModelFidelityReport:
    """Test get_model_fidelity_report tool."""

    def test_report_structure()
    def test_lookback_days_parameter()
    def test_handles_no_validations()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestAssessmentGeneration:
    """Test _generate_assessment function."""

    def test_assessment_excellent_grade()
    def test_assessment_good_grade()
    def test_assessment_fair_grade()
    def test_assessment_poor_grade()
    def test_assessment_with_confounding_factors()
```

**Estimated Tests**: 20 tests

### Dependencies
- Mock `FidelityTracker` and `TwinGenerator` components
- Follow patterns from `test_validity_audit.py`

---

## 3. Performance Regression Tests (Priority 3)

### Current State
- **Documented SLAs**:
  - Drift Monitor: <10s for 50 features
  - Experiment Designer: <60s total latency
  - Health Score: <5s full, <1s quick
- **Gap**: No automated SLA compliance testing

### Implementation

#### File 1: `tests/unit/test_agents/test_drift_monitor/test_performance.py`

```python
"""Performance tests for Drift Monitor Agent - SLA Compliance."""

import pytest
import time

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group(name="performance_tests")]


class TestDriftMonitorSLA:
    """Validate Drift Monitor meets <10s SLA for 50 features."""

    @pytest.mark.asyncio
    async def test_latency_50_features_under_10s()

    @pytest.mark.asyncio
    async def test_latency_10_features_under_3s()

    @pytest.mark.asyncio
    async def test_latency_100_features_under_20s()

    @pytest.mark.asyncio
    async def test_latency_breakdown_by_node()

    @pytest.mark.asyncio
    async def test_latency_with_all_drift_types()


class TestDriftMonitorThroughput:
    """Test throughput characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_drift_checks()

    @pytest.mark.asyncio
    async def test_memory_usage_large_features()
```

**Estimated Tests**: 7 tests

#### File 2: `tests/unit/test_agents/test_experiment_designer/test_performance.py`

```python
"""Performance tests for Experiment Designer Agent - SLA Compliance."""

import pytest
import time

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group(name="performance_tests")]


class TestExperimentDesignerSLA:
    """Validate Experiment Designer meets <60s total latency SLA."""

    @pytest.mark.asyncio
    async def test_total_latency_under_60s()

    @pytest.mark.asyncio
    async def test_twin_simulation_under_2s()

    @pytest.mark.asyncio
    async def test_power_analysis_under_10s()

    @pytest.mark.asyncio
    async def test_validity_audit_under_30s()

    @pytest.mark.asyncio
    async def test_template_generation_under_5s()

    @pytest.mark.asyncio
    async def test_latency_breakdown_by_node()


class TestExperimentDesignerSkipPath:
    """Test skip path performance (early exit on twin SKIP)."""

    @pytest.mark.asyncio
    async def test_skip_path_under_5s()

    @pytest.mark.asyncio
    async def test_skip_avoids_later_nodes()
```

**Estimated Tests**: 8 tests

#### File 3: `tests/unit/test_agents/test_health_score/test_performance.py`

```python
"""Performance tests for Health Score Agent - SLA Compliance."""

import pytest
import time

pytestmark = [pytest.mark.slow, pytest.mark.xdist_group(name="performance_tests")]


class TestHealthScoreSLA:
    """Validate Health Score meets <5s full, <1s quick SLAs."""

    @pytest.mark.asyncio
    async def test_quick_check_under_1s()

    @pytest.mark.asyncio
    async def test_full_check_under_5s()

    @pytest.mark.asyncio
    async def test_models_only_under_2s()

    @pytest.mark.asyncio
    async def test_pipelines_only_under_2s()

    @pytest.mark.asyncio
    async def test_agents_only_under_2s()


class TestHealthScoreParallelism:
    """Test parallel check performance."""

    @pytest.mark.asyncio
    async def test_parallel_checks_faster_than_sequential()

    @pytest.mark.asyncio
    async def test_parallel_check_isolation()

    @pytest.mark.asyncio
    async def test_degraded_component_latency()
```

**Estimated Tests**: 8 tests

### Dependencies
- Uses `@pytest.mark.slow` for optional CI exclusion
- Uses `@pytest.mark.xdist_group(name="performance_tests")` for worker grouping
- No heavy imports - mocks agent dependencies

---

## Summary

### Files to Create/Modify

| File | Action | Tests | Priority |
|------|--------|-------|----------|
| `test_concept_drift.py` | Modify | +21 | P1 |
| `test_simulate_intervention_tool.py` | Create | 20 | P2 |
| `test_validate_twin_fidelity_tool.py` | Create | 20 | P2 |
| `test_drift_monitor/test_performance.py` | Create | 7 | P3 |
| `test_experiment_designer/test_performance.py` | Create | 8 | P3 |
| `test_health_score/test_performance.py` | Create | 8 | P3 |

### Execution Order

```
Wave 1 (P1): Concept Drift Edge Cases
├── Modify test_concept_drift.py
└── Run: pytest tests/unit/test_agents/test_drift_monitor/test_concept_drift.py -v

Wave 2 (P2): Twin Simulation Tool Tests
├── Create test_simulate_intervention_tool.py
├── Create test_validate_twin_fidelity_tool.py
└── Run: pytest tests/unit/test_agents/test_experiment_designer/test_*_tool.py -v

Wave 3 (P3): Performance Regression Tests
├── Create test_drift_monitor/test_performance.py
├── Create test_experiment_designer/test_performance.py
├── Create test_health_score/test_performance.py
└── Run: pytest -m slow -v
```

### Memory Safety Compliance

All new tests follow memory-safe patterns:
- Use `@pytest.mark.xdist_group()` to group related tests on same worker
- Performance tests marked `@pytest.mark.slow` for optional CI exclusion
- Mock heavy components (TwinGenerator, SimulationEngine) instead of importing
- Max 4 workers as per `pyproject.toml` configuration

---

## Acceptance Criteria

1. **Concept Drift**: Test count increases from 6 to ~27
2. **Twin Tools**: Two new test files with ~40 tests total
3. **Performance**: Three new test files with ~23 tests, all marked `@pytest.mark.slow`
4. **All tests pass**: `make test` completes successfully
5. **No memory issues**: Tests run with `-n 4` without OOM

---

## References

- Tier 3 Contracts: `.claude/contracts/tier3-contracts.md`
- Existing test patterns: `tests/unit/test_agents/test_drift_monitor/test_data_drift.py`
- Integration test reference: `tests/integration/test_digital_twin_workflow.py`
- Source files:
  - `src/agents/drift_monitor/nodes/concept_drift.py`
  - `src/agents/experiment_designer/tools/simulate_intervention_tool.py`
  - `src/agents/experiment_designer/tools/validate_twin_fidelity_tool.py`
