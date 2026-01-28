# Plan: Tier 1-5 Agent Testing Framework Using Tier0 Outputs

## Summary

Create a testing framework that uses tier0 synthetic data outputs to validate Tier 1-5 agents for:
- **Agent processing** - Correct execution without errors
- **Output correctness** - Outputs match TypedDict contracts
- **Observability** - Opik traces captured properly

## Current State

- **Tier0 test** (`scripts/run_tier0_test.py`): 8-step MLOps pipeline producing trained models, metrics, cohorts
- **Tier 1-5 agents**: 11 agents across 5 tiers with TypedDict state contracts
- **Opik observability**: Agent-specific tracers with singleton pattern exist

## Tier0 Outputs Available

| Output | Type | Source Step |
|--------|------|-------------|
| `trained_model` | sklearn/xgboost/lightgbm model | Step 5 (Model Trainer) |
| `model_uri` | str (MLflow URI) | Step 5 |
| `validation_metrics` | dict (auc_roc, precision, recall, f1) | Step 5 |
| `feature_importance` | list[{feature, importance}] | Step 6 (Feature Analyzer) |
| `eligible_df` | DataFrame (patient cohort) | Step 3 (Cohort Constructor) |
| `qc_report` | dict (data quality) | Step 2 (Data Preparer) |
| `experiment_id` | str (unique ID) | Step 0 (Scope Definer) |
| `cohort_result` | CohortExecutionResult | Step 3 |
| `scope_spec` | dict (brand, indication, etc.) | Step 0 |
| `class_imbalance_info` | dict (ratio, strategy) | Step 5 |

## Tier 1-5 Agent Input Requirements

### Tier 1: Orchestrator
- `messages`: list[dict] - User query messages
- `current_agent`: Optional[str] - Currently active agent
- `agent_outputs`: dict - Accumulated outputs from delegated agents

### Tier 2: Causal Agents
**CausalImpact**: `treatment_var`, `outcome_var`, `data` (DataFrame), `confounders` (list)
**GapAnalyzer**: `current_metrics`, `target_metrics`, `segment_data` (DataFrame)
**HeterogeneousOptimizer**: `treatment_effects`, `segment_definitions`, `optimization_constraints`

### Tier 3: Monitoring Agents
**DriftMonitor**: `reference_data`, `current_data`, `feature_columns`, `drift_thresholds`
**ExperimentDesigner**: `hypothesis`, `metrics`, `sample_size_params`
**HealthScore**: `system_metrics`, `model_performance`, `data_quality`

### Tier 4: ML Agents
**PredictionSynthesizer**: `model_predictions`, `feature_data`, `ensemble_config`
**ResourceOptimizer**: `resource_constraints`, `optimization_objective`, `segment_valuations`

### Tier 5: Learning Agents
**Explainer**: `model`, `feature_importance`, `prediction_context`
**FeedbackLearner**: `user_feedback`, `agent_outputs`, `improvement_targets`

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/run_tier1_5_test.py` | Main test runner |
| `src/testing/__init__.py` | Testing module init |
| `src/testing/tier0_output_mapper.py` | Maps tier0 outputs to agent inputs |
| `src/testing/contract_validator.py` | TypedDict validation |
| `src/testing/opik_trace_verifier.py` | Observability verification |

## Implementation Details

### 1. Tier0OutputMapper Class

Maps tier0 state dictionary to each agent's required inputs:

```python
class Tier0OutputMapper:
    def __init__(self, tier0_state: Dict[str, Any]):
        self.state = tier0_state
        self._validate_required_keys()

    def map_to_causal_impact(self) -> Dict[str, Any]:
        df = self.state["eligible_df"]
        features = [f["feature"] for f in self.state["feature_importance"][:5]]
        return {
            "treatment_var": self.state["scope_spec"].get("target_variable", "treatment"),
            "outcome_var": self.state["scope_spec"].get("outcome", "outcome"),
            "data": df,
            "confounders": features,
        }

    def map_to_drift_monitor(self) -> Dict[str, Any]:
        df = self.state["eligible_df"]
        split_idx = int(len(df) * 0.8)
        return {
            "reference_data": df.iloc[:split_idx],
            "current_data": df.iloc[split_idx:],
            "feature_columns": [f["feature"] for f in self.state["feature_importance"]],
            "drift_thresholds": {"psi": 0.2, "ks": 0.1},
        }

    def map_to_explainer(self) -> Dict[str, Any]:
        return {
            "model": self.state["trained_model"],
            "feature_importance": self.state["feature_importance"],
            "prediction_context": {
                "brand": self.state["scope_spec"].get("brand", "Unknown"),
                "model_type": type(self.state["trained_model"]).__name__,
            },
        }
    # ... mappers for each agent
```

### 2. ContractValidator Class

Validates agent outputs against TypedDict definitions:

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    checked_fields: int

class ContractValidator:
    def validate_state(self, state: Dict, state_class: type, strict: bool = False) -> ValidationResult:
        hints = get_type_hints(state_class)
        required = getattr(state_class, "__required_keys__", set())
        errors = []
        for field in required:
            if field not in state:
                errors.append(f"Missing required field: {field}")
        return ValidationResult(valid=len(errors) == 0, errors=errors, ...)
```

### 3. OpikTraceVerifier Class

Verifies Opik traces were created:

```python
class OpikTraceVerifier:
    def __init__(self, opik_base_url: str = "http://localhost:5173"):
        self.base_url = opik_base_url

    async def verify_trace_exists(self, trace_id: str) -> bool:
        resp = await self.client.get(f"{self.base_url}/api/traces/{trace_id}")
        return resp.status_code == 200

    async def verify_agent_trace(self, agent_name: str, trace_id: str, tier: int) -> ValidationResult:
        expected = {"agent_name": agent_name, "tier": tier, "framework": "langgraph"}
        return await self.verify_trace_metadata(trace_id, expected)
```

### 4. Test Runner Structure

```python
AGENT_CONFIGS = {
    "orchestrator": {"tier": 1, "state_class": "OrchestratorState"},
    "causal_impact": {"tier": 2, "state_class": "CausalImpactState"},
    "gap_analyzer": {"tier": 2, "state_class": "GapAnalyzerState"},
    "drift_monitor": {"tier": 3, "state_class": "DriftMonitorState"},
    "explainer": {"tier": 5, "state_class": "ExplainerState"},
    # ... all 11 agents
}

async def test_agent(agent_name, config, mapper, validator, trace_verifier):
    # 1. Get mapped inputs
    agent_input = getattr(mapper, f"map_to_{agent_name}")()
    # 2. Run agent
    agent = get_agent_class(agent_name)()
    output = await agent.run(agent_input)
    # 3. Validate contract
    contract_result = validator.validate_state(output, state_class)
    # 4. Verify observability
    trace_result = await trace_verifier.verify_agent_trace(...)
    return AgentTestResult(...)
```

## Test Execution

```bash
# On droplet - Run tier0 first, then test all Tier 1-5 agents
python scripts/run_tier1_5_test.py --run-tier0-first

# Use cached tier0 outputs (faster iteration)
python scripts/run_tier1_5_test.py --tier0-cache scripts/tier0_output_cache/latest.pkl

# Test specific tiers
python scripts/run_tier1_5_test.py --tiers 2,3

# Test specific agents
python scripts/run_tier1_5_test.py --agents causal_impact,explainer

# Skip Opik verification (if Opik not running)
python scripts/run_tier1_5_test.py --skip-observability

# Save results to JSON
python scripts/run_tier1_5_test.py --output results/tier1_5_test_results.json
```

## Detailed Result Structures

### AgentTestResult Dataclass

Each agent test captures comprehensive details for review:

```python
@dataclass
class AgentTestResult:
    """Complete result of testing a single agent."""
    # Identity
    agent_name: str
    tier: int
    test_timestamp: str  # ISO format

    # Execution
    success: bool
    execution_time_ms: float
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    # Input Summary
    input_summary: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"data_rows": 500, "features_count": 10, "treatment_var": "therapy_switch"}

    # Agent Output (full)
    agent_output: Optional[Dict[str, Any]] = None

    # Contract Validation Details
    contract_validation: Optional[ContractValidationDetail] = None

    # Observability Details
    trace_verification: Optional[TraceVerificationDetail] = None

    # Performance Metrics
    performance_metrics: Optional[PerformanceDetail] = None

@dataclass
class ContractValidationDetail:
    """Detailed contract validation results."""
    valid: bool
    state_class: str
    required_fields_checked: list[str]
    optional_fields_checked: list[str]
    missing_required: list[str]
    type_errors: list[Dict[str, str]]  # [{"field": "x", "expected": "str", "actual": "int"}]
    extra_fields: list[str]  # Fields not in TypedDict
    warnings: list[str]

@dataclass
class TraceVerificationDetail:
    """Detailed Opik trace verification results."""
    trace_exists: bool
    trace_id: Optional[str]
    trace_url: Optional[str]  # Direct link to Opik UI
    metadata_valid: bool
    expected_metadata: Dict[str, Any]
    actual_metadata: Dict[str, Any]
    span_count: int
    span_names: list[str]
    duration_ms: Optional[float]

@dataclass
class PerformanceDetail:
    """Agent performance metrics."""
    total_time_ms: float
    llm_calls: int
    llm_tokens_input: int
    llm_tokens_output: int
    tool_calls: int
    memory_peak_mb: Optional[float]
```

### Detailed Console Output Per Agent

```
============================================================
TESTING: causal_impact (Tier 2)
============================================================

  INPUT SUMMARY:
    treatment_var: therapy_switch
    outcome_var: discontinuation
    data_rows: 500
    data_columns: 12
    confounders: ['age', 'prior_lines', 'ecog_score', 'time_on_therapy', 'adverse_events']

  EXECUTION:
    Status: PASS
    Time: 1523.2ms
    LLM Calls: 3
    Tool Calls: 2 (identify_confounders, estimate_effect)

  AGENT OUTPUT:
    causal_effect:
      ate: 0.127
      ate_ci: [0.089, 0.165]
      p_value: 0.0023
    confounders_identified: ['age', 'prior_lines', 'ecog_score']
    causal_graph:
      nodes: 6
      edges: 8
    recommendations:
      - "Therapy switch shows significant positive effect on discontinuation"
      - "Age and prior_lines are key confounders"

  CONTRACT VALIDATION:
    State Class: CausalImpactOutputState
    Required Fields: 5/5 present
      ✓ causal_effect (dict)
      ✓ confounders_identified (list)
      ✓ causal_graph (dict)
      ✓ analysis_complete (bool)
      ✓ trace_id (str)
    Optional Fields: 2/3 present
      ✓ recommendations (list)
      ✓ sensitivity_analysis (dict)
      - effect_modifiers (not provided)
    Type Errors: 0
    Warnings: 0

  OBSERVABILITY:
    Trace ID: 019462ab-7c3d-7f8a-9b2e-1234567890ab
    Trace URL: http://localhost:5173/traces/019462ab-7c3d-7f8a-9b2e-1234567890ab
    Metadata Valid: True
      ✓ agent_name: causal_impact
      ✓ tier: 2
      ✓ framework: langgraph
      ✓ experiment_id: tier0_test_abc123
    Spans: 5
      - causal_impact.run (1523ms)
      - identify_confounders (234ms)
      - build_causal_graph (456ms)
      - estimate_effect (678ms)
      - generate_recommendations (155ms)

  ✅ PASSED
```

### Failed Agent Example

```
============================================================
TESTING: feedback_learner (Tier 5)
============================================================

  INPUT SUMMARY:
    user_feedback_count: 5
    agent_outputs_keys: ['causal_impact', 'gap_analyzer']
    improvement_targets: ['accuracy', 'explanation_clarity']

  EXECUTION:
    Status: FAIL
    Time: 2341.5ms
    Error: ValidationError
    Traceback:
      File "src/agents/feedback_learner/agent.py", line 89, in run
        improvement_plan = await self._generate_plan(feedback)
      File "src/agents/feedback_learner/agent.py", line 134, in _generate_plan
        raise ValidationError("Insufficient feedback items for learning")

  AGENT OUTPUT: None (execution failed)

  CONTRACT VALIDATION:
    State Class: FeedbackLearnerOutputState
    Required Fields: 0/4 present (execution failed before output)
      ✗ improvement_plan (missing)
      ✗ learned_patterns (missing)
      ✗ confidence_scores (missing)
      ✗ trace_id (missing)

  OBSERVABILITY:
    Trace ID: 019462ab-8d4e-7f8a-9b2e-0987654321ba
    Trace URL: http://localhost:5173/traces/019462ab-8d4e-7f8a-9b2e-0987654321ba
    Metadata Valid: True
    Spans: 2
      - feedback_learner.run (2341ms) [ERROR]
      - _generate_plan (1892ms) [ERROR]
    Error Captured in Trace: Yes

  ❌ FAILED: ValidationError - Insufficient feedback items for learning
```

### JSON Report Structure

Full results saved to `--output` file for programmatic review:

```json
{
  "test_run": {
    "id": "tier1_5_test_20250127_201534",
    "timestamp": "2025-01-27T20:15:34.123Z",
    "tier0_cache": "scripts/tier0_output_cache/latest.pkl",
    "tier0_experiment_id": "tier0_test_abc123"
  },
  "summary": {
    "total_agents": 11,
    "passed": 10,
    "failed": 1,
    "skipped": 0,
    "total_time_ms": 15234.5,
    "pass_rate": 0.909
  },
  "tier_breakdown": {
    "tier_1": {"passed": 2, "failed": 0, "agents": ["orchestrator", "tool_composer"]},
    "tier_2": {"passed": 3, "failed": 0, "agents": ["causal_impact", "gap_analyzer", "heterogeneous_optimizer"]},
    "tier_3": {"passed": 3, "failed": 0, "agents": ["drift_monitor", "experiment_designer", "health_score"]},
    "tier_4": {"passed": 2, "failed": 0, "agents": ["prediction_synthesizer", "resource_optimizer"]},
    "tier_5": {"passed": 1, "failed": 1, "agents": ["explainer", "feedback_learner"]}
  },
  "results": [
    {
      "agent_name": "causal_impact",
      "tier": 2,
      "test_timestamp": "2025-01-27T20:15:36.456Z",
      "success": true,
      "execution_time_ms": 1523.2,
      "input_summary": {
        "treatment_var": "therapy_switch",
        "outcome_var": "discontinuation",
        "data_rows": 500,
        "confounders_count": 5
      },
      "agent_output": {
        "causal_effect": {"ate": 0.127, "ate_ci": [0.089, 0.165], "p_value": 0.0023},
        "confounders_identified": ["age", "prior_lines", "ecog_score"],
        "causal_graph": {"nodes": 6, "edges": 8},
        "analysis_complete": true,
        "trace_id": "019462ab-7c3d-7f8a-9b2e-1234567890ab"
      },
      "contract_validation": {
        "valid": true,
        "state_class": "CausalImpactOutputState",
        "required_fields_checked": ["causal_effect", "confounders_identified", "causal_graph", "analysis_complete", "trace_id"],
        "missing_required": [],
        "type_errors": [],
        "warnings": []
      },
      "trace_verification": {
        "trace_exists": true,
        "trace_id": "019462ab-7c3d-7f8a-9b2e-1234567890ab",
        "trace_url": "http://localhost:5173/traces/019462ab-7c3d-7f8a-9b2e-1234567890ab",
        "metadata_valid": true,
        "span_count": 5,
        "span_names": ["causal_impact.run", "identify_confounders", "build_causal_graph", "estimate_effect", "generate_recommendations"]
      },
      "performance_metrics": {
        "total_time_ms": 1523.2,
        "llm_calls": 3,
        "llm_tokens_input": 4521,
        "llm_tokens_output": 892,
        "tool_calls": 2
      }
    },
    {
      "agent_name": "feedback_learner",
      "tier": 5,
      "test_timestamp": "2025-01-27T20:16:12.789Z",
      "success": false,
      "execution_time_ms": 2341.5,
      "error": "ValidationError: Insufficient feedback items for learning",
      "error_traceback": "File \"src/agents/feedback_learner/agent.py\", line 89...",
      "input_summary": {
        "user_feedback_count": 5,
        "agent_outputs_keys": ["causal_impact", "gap_analyzer"],
        "improvement_targets": ["accuracy", "explanation_clarity"]
      },
      "agent_output": null,
      "contract_validation": {
        "valid": false,
        "missing_required": ["improvement_plan", "learned_patterns", "confidence_scores", "trace_id"]
      },
      "trace_verification": {
        "trace_exists": true,
        "trace_id": "019462ab-8d4e-7f8a-9b2e-0987654321ba",
        "error_captured": true
      }
    }
  ],
  "observability_summary": {
    "traces_created": 11,
    "traces_verified": 11,
    "traces_with_errors": 1,
    "total_spans": 47,
    "opik_health": "healthy"
  }
}
```

### Review Commands

```bash
# View full JSON report
cat results/tier1_5_test_results.json | python -m json.tool

# Extract failed agents only
cat results/tier1_5_test_results.json | jq '.results[] | select(.success == false)'

# Get tier-by-tier summary
cat results/tier1_5_test_results.json | jq '.tier_breakdown'

# List all agent outputs
cat results/tier1_5_test_results.json | jq '.results[] | {agent: .agent_name, output: .agent_output}'

# Find contract validation errors
cat results/tier1_5_test_results.json | jq '.results[] | select(.contract_validation.valid == false) | {agent: .agent_name, errors: .contract_validation}'

# Open Opik traces for failed agents (on droplet)
cat results/tier1_5_test_results.json | jq -r '.results[] | select(.success == false) | .trace_verification.trace_url'
```

## Expected Console Output

```
============================================================
TIER 1-5 AGENT TESTING FRAMEWORK
============================================================
Tier0 Cache: scripts/tier0_output_cache/latest.pkl
Experiment ID: tier0_test_abc123
Agents to Test: 11

Loading tier0 state...
  trained_model: XGBClassifier
  eligible_df: 500 rows x 12 columns
  feature_importance: 10 features
  validation_metrics: {auc_roc: 0.87, precision: 0.82, recall: 0.79}

============================================================
TESTING: orchestrator (Tier 1)
============================================================

  INPUT SUMMARY:
    messages: 1 (test query: "What factors drive therapy discontinuation?")
    current_agent: None
    agent_outputs: {}

  EXECUTION:
    Status: PASS
    Time: 892.3ms
    ...

[... detailed output for each agent ...]

============================================================
TEST SUMMARY
============================================================

TIER RESULTS:
  Tier 1: 2/2 PASSED (orchestrator, tool_composer)
  Tier 2: 3/3 PASSED (causal_impact, gap_analyzer, heterogeneous_optimizer)
  Tier 3: 3/3 PASSED (drift_monitor, experiment_designer, health_score)
  Tier 4: 2/2 PASSED (prediction_synthesizer, resource_optimizer)
  Tier 5: 1/2 PASSED (explainer passed, feedback_learner FAILED)

OVERALL:
  Total: 11
  Passed: 10
  Failed: 1
  Skipped: 0
  Pass Rate: 90.9%
  Total Time: 15234.5ms

FAILED AGENTS:
  - feedback_learner (Tier 5): ValidationError - Insufficient feedback items

OBSERVABILITY:
  Traces Created: 11
  Traces Verified: 11
  Opik Dashboard: http://localhost:5173/projects/tier1_5_test

Results saved to: results/tier1_5_test_results.json
```

## Verification Criteria

| Criterion | How Verified |
|-----------|--------------|
| **Agent Processing** | No exceptions, completes within 30s timeout |
| **Output Correctness** | TypedDict contract validation (required fields, types) |
| **Observability** | Opik trace exists with correct metadata (agent_name, tier) |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Missing tier0 outputs | Mapper validates required keys upfront |
| Agent import failures | Try/except with informative error messages |
| Opik not running | `--skip-observability` flag available |
| Long test times | Timeout per agent (30s), can run specific tiers |

## Constraints Respected

- No pip install on droplet - uses existing venv
- Uses real tier0 outputs (not mocks)
- Integrates with existing Opik infrastructure
- Follows existing test patterns from run_tier0_test.py
- Minimal new dependencies (uses stdlib + existing packages)
