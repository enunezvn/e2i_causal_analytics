# Plan: Fix Tier 1-5 Test Framework Quality Control

## Problem Summary

The tier1-5 test framework reports **100% pass rate** when agents actually fail:
- `tool_composer`: Returns `success: False` with error
- `gap_analyzer`: Returns error message in executive_summary
- `prediction_synthesizer`: Returns `status: failed`
- `drift_monitor`: Returns all empty arrays `[]`
- Most agents missing 50-80% of required fields

**Root Causes:**
1. ContractValidator uses "lenient" mode → missing fields become warnings, not errors
2. Test success criteria only checks for exceptions, not data quality
3. Agents silently fall back to mock data when Supabase unavailable
4. No agent-specific quality thresholds

---

## Solution: Per-Agent Quality Gate System

### Design Principles
1. **No lenient mode** - agents either pass or fail
2. **Agent-specific thresholds** - each agent has defined success criteria
3. **Data quality validation** - check for meaningful output, not just structure
4. **Explicit failure modes** - surface errors clearly, don't hide them

---

## Implementation Plan

### Step 1: Create Agent Quality Gate Configuration

**File**: `src/testing/agent_quality_gates.py` (NEW)

Define per-agent quality thresholds:

```python
AGENT_QUALITY_GATES = {
    "orchestrator": {
        "required_output_fields": ["status", "response_text", "agents_dispatched"],
        "min_required_fields_pct": 0.8,  # 80% of contract required fields
        "data_quality_checks": {
            "agents_dispatched": {"type": "list", "min_length": 1},
            "response_confidence": {"type": "float", "min_value": 0.5},
        },
        "fail_on_status": ["error", "failed"],
    },
    "tool_composer": {
        "required_output_fields": ["success", "composition_id"],
        "min_required_fields_pct": 0.7,
        "data_quality_checks": {
            "success": {"type": "bool", "must_be": True},
            "tools_executed": {"type": "int", "min_value": 1},
        },
        "fail_on_status": ["failed"],
    },
    "causal_impact": {
        "required_output_fields": ["status", "ate_estimate", "executive_summary"],
        "min_required_fields_pct": 0.6,
        "data_quality_checks": {
            "ate_estimate": {"type": "float", "not_null": True},
            "confidence_interval": {"type": "tuple", "not_null": True},
        },
        "fail_on_status": ["error", "failed"],
    },
    "gap_analyzer": {
        "required_output_fields": ["executive_summary", "prioritized_opportunities"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "executive_summary": {"type": "str", "not_contains": ["Error:", "error"]},
            "prioritized_opportunities": {"type": "list"},  # Can be empty if no gaps
        },
        "fail_on_status": ["error", "failed"],
    },
    "heterogeneous_optimizer": {
        "required_output_fields": ["overall_ate", "heterogeneity_score", "status"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "overall_ate": {"type": "float", "not_null": True},
            "heterogeneity_score": {"type": "float", "min_value": 0.0},
        },
        "fail_on_status": ["error", "failed"],
    },
    "drift_monitor": {
        "required_output_fields": ["overall_drift_score", "data_drift_results"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "overall_drift_score": {"type": "float", "not_null": True},
            # Note: drift_results can be empty if no drift detected
        },
        "fail_on_status": ["error", "failed"],
    },
    "experiment_designer": {
        "required_output_fields": ["design_type", "treatments", "outcomes"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "design_type": {"type": "str", "not_null": True},
            "treatments": {"type": "list", "min_length": 1},
        },
        "fail_on_status": ["error", "failed"],
    },
    "health_score": {
        "required_output_fields": ["overall_health_score", "health_grade"],
        "min_required_fields_pct": 0.6,
        "data_quality_checks": {
            "overall_health_score": {"type": "float", "min_value": 0.0, "max_value": 100.0},
            "health_grade": {"type": "str", "in_set": ["A", "B", "C", "D", "F"]},
        },
        "fail_on_status": ["error", "failed"],
    },
    "prediction_synthesizer": {
        "required_output_fields": ["status", "ensemble_prediction"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "status": {"type": "str", "must_not_be": "failed"},
            "models_succeeded": {"type": "int", "min_value": 1},
        },
        "fail_on_status": ["failed", "error"],
    },
    "resource_optimizer": {
        "required_output_fields": ["status", "optimal_allocations", "solver_status"],
        "min_required_fields_pct": 0.6,
        "data_quality_checks": {
            "solver_status": {"type": "str", "in_set": ["optimal", "feasible"]},
            "optimal_allocations": {"type": "list", "min_length": 1},
        },
        "fail_on_status": ["error", "failed", "infeasible"],
    },
    "explainer": {
        "required_output_fields": ["executive_summary", "status"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "executive_summary": {"type": "str", "min_length": 20},
        },
        "fail_on_status": ["error", "failed"],
    },
    "feedback_learner": {
        "required_output_fields": ["learning_summary", "status"],
        "min_required_fields_pct": 0.5,
        "data_quality_checks": {
            "status": {"type": "str", "in_set": ["completed", "partial"]},
        },
        "fail_on_status": ["error", "failed"],
    },
}
```

### Step 2: Create Quality Gate Validator

**File**: `src/testing/quality_gate_validator.py` (NEW)

```python
class QualityGateValidator:
    """Validates agent outputs against per-agent quality gates."""

    def validate(self, agent_name: str, output: dict) -> QualityGateResult:
        """
        Returns:
            QualityGateResult with:
            - passed: bool
            - failed_checks: list of specific failures
            - warnings: list of non-critical issues
        """
```

### Step 3: Update ContractValidator - Remove Lenient Mode

**File**: `src/testing/contract_validator.py`

Changes:
1. Remove `lenient` parameter from constructor
2. Missing required fields are ALWAYS errors
3. Add `strict_mode` that fails on ANY issue (optional for debugging)

### Step 4: Update Test Framework Success Logic

**File**: `scripts/run_tier1_5_test.py`

Current (BROKEN):
```python
validator = ContractValidator(lenient=True)  # Line 1324
result.success = True
if result.contract_validation and not result.contract_validation.valid:
    if result.contract_validation.missing_required:
        result.success = False
```

New (FIXED):
```python
from src.testing.agent_quality_gates import AGENT_QUALITY_GATES
from src.testing.quality_gate_validator import QualityGateValidator

validator = ContractValidator()  # NO lenient mode
quality_validator = QualityGateValidator(AGENT_QUALITY_GATES)

# After agent execution:
quality_result = quality_validator.validate(agent_name, result.agent_output)
result.quality_gate = quality_result

# Success requires ALL of:
# 1. No execution errors
# 2. Contract validation passes (required fields present)
# 3. Quality gate passes (data quality checks)
result.success = (
    result.error is None
    and result.contract_validation.valid
    and result.quality_gate.passed
)
```

### Step 5: Enhanced Output Reporting

Update `print_agent_result()` to show quality gate results:

```
📊 Quality Gate:
  ✅ Required output fields: 3/3 present
  ✅ Data quality checks: 4/4 passed
  ❌ Status check: FAIL - status="failed" is in fail_on_status list

  QUALITY GATE: ❌ FAIL
```

---

## Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `src/testing/agent_quality_gates.py` | CREATE | Per-agent threshold definitions |
| `src/testing/quality_gate_validator.py` | CREATE | Quality gate validation logic |
| `src/testing/contract_validator.py` | MODIFY | Remove lenient mode |
| `src/testing/__init__.py` | MODIFY | Export new classes |
| `scripts/run_tier1_5_test.py` | MODIFY | Integrate quality gates, fix success logic |

---

## Verification

After implementation, re-run the tier1-5 test:

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && \
   /opt/e2i_causal_analytics/.venv/bin/python scripts/run_tier1_5_test.py \
   --skip-observability --timeout 60"
```

Expected results with proper QC:
- `tool_composer`: ❌ FAIL (success=False)
- `gap_analyzer`: ❌ FAIL (error in executive_summary)
- `prediction_synthesizer`: ❌ FAIL (status=failed)
- `drift_monitor`: ⚠️ PASS with warnings (empty results but no error)
- Others: Varies based on actual data quality

---

## Success Criteria

1. Agents that return errors should FAIL, not PASS
2. Each agent has documented, reviewable quality thresholds
3. Test output clearly shows WHY an agent failed
4. No "lenient" mode that hides real problems
