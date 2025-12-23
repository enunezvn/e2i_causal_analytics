# Tier 4 Contracts: ML Predictions Agents

**Version**: 1.0
**Last Updated**: 2025-12-18
**Status**: Active

## Overview

This document defines integration contracts for **Tier 4: ML Predictions** agents in the E2I Causal Analytics platform. These agents handle ML prediction synthesis and resource allocation optimization.

### Tier 4 Agents

| Agent | Type | Responsibility | Primary Methods |
|-------|------|----------------|-----------------|
| **Prediction Synthesizer** | Standard (Computational) | ML prediction aggregation and ensemble | Model orchestration, ensemble methods |
| **Resource Optimizer** | Standard (Computational) | Resource allocation optimization | Linear/MILP/nonlinear optimization |

---

## 1. Shared Types

### 1.1 Common Enums

```python
from typing import Literal

# Ensemble methods
EnsembleMethod = Literal["average", "weighted", "stacking", "voting"]

# Optimization objectives
OptimizationObjective = Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"]

# Solver types
SolverType = Literal["linear", "milp", "nonlinear"]

# Trend directions
TrendDirection = Literal["increasing", "stable", "decreasing"]

# Entity types
EntityType = Literal["hcp", "territory", "region", "patient"]
```

### 1.2 Common Input Fields

All Tier 4 agents accept these common fields:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Tier4CommonInput(BaseModel):
    """Common input fields for all Tier 4 agents"""
    query: str = Field(..., description="User's natural language query")
```

### 1.3 Common Output Fields

All Tier 4 agents return these common fields:

```python
from typing import List

class Tier4CommonOutput(BaseModel):
    """Common output fields for all Tier 4 agents"""
    total_latency_ms: int = Field(..., description="Total processing time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    timestamp: str = Field(..., description="ISO 8601 timestamp of completion")
```

---

## 2. Prediction Synthesizer Agent

**Agent Type**: Standard (Computational)
**Primary Methods**: Model orchestration, ensemble methods
**Latency**: Up to 15s

### 2.1 Input Contract

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class PredictionSynthesizerInput(BaseModel):
    """Input contract for Prediction Synthesizer Agent"""

    # Required fields
    query: str
    entity_id: str = Field(..., description="ID of entity to predict for")
    entity_type: EntityType = Field(..., description="Type of entity")
    prediction_target: str = Field(..., description="What to predict (e.g., 'trx', 'churn_risk')")
    features: Dict[str, Any] = Field(..., description="Input features for prediction")

    # Optional fields
    time_horizon: str = Field("30d", description="Time horizon for prediction (e.g., '30d', '90d')")
    models_to_use: Optional[List[str]] = Field(
        None,
        description="Specific models to use, or None for all applicable"
    )

    # Configuration
    ensemble_method: EnsembleMethod = Field("weighted", description="How to combine predictions")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level for intervals")
    include_context: bool = Field(True, description="Include prediction context and explanations")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the predicted TRx for this HCP in the next 30 days?",
                "entity_id": "hcp_12345",
                "entity_type": "hcp",
                "prediction_target": "trx",
                "features": {
                    "hcp_specialty": "Oncology",
                    "patient_volume_decile": 9,
                    "current_trx": 45,
                    "engagement_frequency": 2.5
                },
                "time_horizon": "30d",
                "ensemble_method": "weighted",
                "include_context": True
            }
        }
    }
```

### 2.2 Output Contract

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelPrediction(BaseModel):
    """Individual model prediction"""
    model_id: str
    model_type: str
    prediction: float
    prediction_proba: Optional[List[float]] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: int
    features_used: List[str]

class EnsemblePrediction(BaseModel):
    """Combined ensemble prediction"""
    point_estimate: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    ensemble_method: EnsembleMethod
    model_agreement: float = Field(..., ge=0.0, le=1.0, description="Model agreement score")

class PredictionContext(BaseModel):
    """Context for interpreting prediction"""
    similar_cases: List[Dict[str, Any]] = Field(..., description="Similar historical cases")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance rankings")
    historical_accuracy: float = Field(..., ge=0.0, le=1.0, description="Historical model accuracy")
    trend_direction: TrendDirection

class PredictionSynthesizerOutput(BaseModel):
    """Output contract for Prediction Synthesizer Agent"""

    # Core prediction
    ensemble_prediction: EnsemblePrediction = Field(..., description="Combined ensemble prediction")
    prediction_summary: str = Field(..., description="Human-readable summary")

    # Individual models
    individual_predictions: List[ModelPrediction] = Field(..., description="All model predictions")
    models_succeeded: int = Field(..., description="Number of successful predictions")
    models_failed: int = Field(..., description="Number of failed predictions")

    # Context
    prediction_context: Optional[PredictionContext] = Field(None, description="Prediction context if requested")

    # Metadata
    orchestration_latency_ms: int
    ensemble_latency_ms: int
    total_latency_ms: int
    timestamp: str
    warnings: List[str] = Field(default_factory=list)
```

### 2.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class PredictionSynthesizerState(TypedDict):
    """Complete LangGraph state for Prediction Synthesizer Agent"""

    # === INPUT ===
    query: str
    entity_id: str
    entity_type: str
    prediction_target: str
    features: Dict[str, Any]
    time_horizon: str

    # === CONFIGURATION ===
    models_to_use: Optional[List[str]]
    ensemble_method: Literal["average", "weighted", "stacking", "voting"]
    confidence_level: float
    include_context: bool

    # === MODEL OUTPUTS ===
    individual_predictions: Optional[List[Dict[str, Any]]]  # ModelPrediction
    models_succeeded: int
    models_failed: int

    # === ENSEMBLE OUTPUTS ===
    ensemble_prediction: Optional[Dict[str, Any]]  # EnsemblePrediction
    prediction_summary: Optional[str]

    # === CONTEXT OUTPUTS ===
    prediction_context: Optional[Dict[str, Any]]  # PredictionContext

    # === EXECUTION METADATA ===
    orchestration_latency_ms: int
    ensemble_latency_ms: int
    total_latency_ms: int
    timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "predicting", "combining", "enriching", "completed", "failed"]
```

### 2.4 Handoff Format

```yaml
prediction_synthesizer_handoff:
  agent: prediction_synthesizer
  analysis_type: prediction
  status: completed

  key_findings:
    prediction: 52.3
    confidence_interval: [48.1, 56.5]
    confidence: 0.82
    model_agreement: 0.91
    ensemble_method: weighted

  models:
    succeeded: 3
    failed: 0
    used:
      - model_id: "trx_predictor_v2"
        prediction: 51.8
        confidence: 0.85
      - model_id: "trx_gradient_boost"
        prediction: 52.5
        confidence: 0.83
      - model_id: "trx_neural_net"
        prediction: 52.7
        confidence: 0.78

  context:
    trend: increasing
    historical_accuracy: 0.87
    top_features:
      patient_volume_decile: 0.32
      hcp_specialty: 0.28
      engagement_frequency: 0.21
    similar_cases: 5

  summary: "Prediction: 52.3 (95% CI: [48.1, 56.5]). Confidence: high. Model agreement: strong across 3 models."

  recommendations:
    - "High confidence prediction - suitable for decision-making"
    - "All models agree within 2% - low prediction uncertainty"
    - "Increasing trend suggests continued growth"

  warnings:
    - "Historical accuracy measured on last 90 days only"

  requires_further_analysis: false
  suggested_next_agent: explainer
  suggested_reason: "Generate explanation for stakeholders"
```

---

## 3. Resource Optimizer Agent

**Agent Type**: Standard (Computational)
**Primary Methods**: Linear/MILP/nonlinear optimization
**Latency**: Up to 20s

### 3.1 Input Contract

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AllocationTarget(BaseModel):
    """Target entity for allocation"""
    entity_id: str
    entity_type: EntityType
    current_allocation: float = Field(..., ge=0)
    min_allocation: Optional[float] = Field(None, ge=0)
    max_allocation: Optional[float] = None
    expected_response: float = Field(..., description="Response coefficient per unit allocation")

class Constraint(BaseModel):
    """Optimization constraint"""
    constraint_type: str = Field(..., description="budget|capacity|min_coverage|max_frequency")
    value: float
    scope: str = Field("global", description="global|regional|entity")

class ResourceOptimizerInput(BaseModel):
    """Input contract for Resource Optimizer Agent"""

    # Required fields
    query: str
    resource_type: str = Field(..., description="budget|rep_time|samples|calls")
    allocation_targets: List[AllocationTarget] = Field(..., min_items=1, description="Entities to allocate to")
    constraints: List[Constraint] = Field(..., min_items=1, description="Optimization constraints")

    # Optional fields
    objective: OptimizationObjective = Field("maximize_outcome", description="Optimization objective")

    # Configuration
    solver_type: Optional[SolverType] = Field(None, description="Solver type, or None for auto-select")
    time_limit_seconds: int = Field(10, ge=1, le=60, description="Max solver time")
    gap_tolerance: float = Field(0.01, ge=0, le=0.1, description="MILP gap tolerance")
    run_scenarios: bool = Field(False, description="Run scenario analysis")
    scenario_count: int = Field(3, ge=1, le=10, description="Number of scenarios")

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How should we allocate budget across territories?",
                "resource_type": "budget",
                "allocation_targets": [
                    {
                        "entity_id": "territory_001",
                        "entity_type": "territory",
                        "current_allocation": 50000,
                        "min_allocation": 30000,
                        "max_allocation": 100000,
                        "expected_response": 0.05
                    },
                    {
                        "entity_id": "territory_002",
                        "entity_type": "territory",
                        "current_allocation": 40000,
                        "min_allocation": 25000,
                        "max_allocation": 80000,
                        "expected_response": 0.08
                    }
                ],
                "constraints": [
                    {
                        "constraint_type": "budget",
                        "value": 200000,
                        "scope": "global"
                    }
                ],
                "objective": "maximize_outcome"
            }
        }
    }
```

### 3.2 Output Contract

```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class AllocationResult(BaseModel):
    """Optimized allocation for an entity"""
    entity_id: str
    entity_type: EntityType
    current_allocation: float
    optimized_allocation: float
    change: float
    change_percentage: float
    expected_impact: float

class ScenarioResult(BaseModel):
    """Result of a scenario analysis"""
    scenario_name: str
    total_allocation: float
    projected_outcome: float
    roi: float
    constraint_violations: List[str] = Field(default_factory=list)

class ResourceOptimizerOutput(BaseModel):
    """Output contract for Resource Optimizer Agent"""

    # Core optimization results
    optimal_allocations: List[AllocationResult] = Field(..., description="Optimized allocations")
    objective_value: float = Field(..., description="Objective function value at optimum")
    solver_status: str = Field(..., description="optimal|infeasible|failed")

    # Impact projections
    projected_total_outcome: float = Field(..., description="Projected total outcome")
    projected_roi: float = Field(..., ge=0, description="Return on investment")
    impact_by_segment: Dict[str, float] = Field(..., description="Impact by entity type")

    # Scenario analysis (if requested)
    scenarios: Optional[List[ScenarioResult]] = Field(None, description="Scenario analysis results")
    sensitivity_analysis: Optional[Dict[str, float]] = Field(None, description="Sensitivity to parameters")

    # Summary
    optimization_summary: str = Field(..., description="Human-readable summary")
    recommendations: List[str] = Field(..., description="Actionable recommendations")

    # Metadata
    formulation_latency_ms: int
    optimization_latency_ms: int
    solve_time_ms: int
    total_latency_ms: int
    timestamp: str
    warnings: List[str] = Field(default_factory=list)
```

### 3.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class ResourceOptimizerState(TypedDict):
    """Complete LangGraph state for Resource Optimizer Agent"""

    # === INPUT ===
    query: str
    resource_type: str
    allocation_targets: List[Dict[str, Any]]  # AllocationTarget
    constraints: List[Dict[str, Any]]  # Constraint
    objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"]

    # === CONFIGURATION ===
    solver_type: Literal["linear", "milp", "nonlinear"]
    time_limit_seconds: int
    gap_tolerance: float
    run_scenarios: bool
    scenario_count: int

    # === OPTIMIZATION OUTPUTS ===
    optimal_allocations: Optional[List[Dict[str, Any]]]  # AllocationResult
    objective_value: Optional[float]
    solver_status: Optional[str]
    solve_time_ms: int

    # === SCENARIO OUTPUTS ===
    scenarios: Optional[List[Dict[str, Any]]]  # ScenarioResult
    sensitivity_analysis: Optional[Dict[str, float]]

    # === IMPACT OUTPUTS ===
    projected_total_outcome: Optional[float]
    projected_roi: Optional[float]
    impact_by_segment: Optional[Dict[str, float]]

    # === SUMMARY ===
    optimization_summary: Optional[str]
    recommendations: Optional[List[str]]

    # === EXECUTION METADATA ===
    formulation_latency_ms: int
    optimization_latency_ms: int
    total_latency_ms: int

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "formulating", "optimizing", "analyzing", "projecting", "completed", "failed"]
```

### 3.4 Handoff Format

```yaml
resource_optimizer_handoff:
  agent: resource_optimizer
  analysis_type: resource_optimization
  status: completed

  key_findings:
    objective_value: 12500
    projected_outcome: 12500
    projected_roi: 2.08
    solver_status: optimal
    solve_time_ms: 245

  allocations:
    total_entities: 15
    increases: 8
    decreases: 7
    unchanged: 0
    top_change:
      entity_id: "territory_005"
      change: +45000
      change_percentage: +90.0
      expected_impact: +2250

  impact_by_segment:
    territory: 8500
    region: 4000

  top_recommendations:
    - "Increase allocation to territory_005 by 45000 (+90%) - Expected impact: 2250"
    - "Increase allocation to territory_012 by 35000 (+70%) - Expected impact: 2100"
    - "Increase allocation to territory_008 by 28000 (+56%) - Expected impact: 1680"
    - "Reduce allocation from territory_003 by 20000 (-40%) - Reallocate to higher-impact targets"
    - "Reduce allocation from territory_007 by 15000 (-30%) - Reallocate to higher-impact targets"

  summary: "Optimization complete. Projected outcome: 12500 (ROI: 2.08). Recommended changes: 8 increases, 7 decreases."

  constraints_satisfied:
    - "Budget constraint: 200000 (100% utilized)"
    - "All min/max allocation bounds respected"

  warnings:
    - "Expected response coefficients based on historical data - may not reflect future behavior"

  requires_further_analysis: false
  suggested_next_agent: gap_analyzer
  suggested_reason: "Validate optimization against actual gap opportunities"
```

---

## 4. Inter-Agent Communication

### 4.1 Orchestrator → Tier 4 Dispatch

All Tier 4 agents receive dispatches via the standard `AgentDispatchRequest` (see `orchestrator-contracts.md`).

### 4.2 Tier 4 → Orchestrator Response

All Tier 4 agents return via the standard `AgentDispatchResponse`.

### 4.3 Tier 4 Inter-Agent Handoffs

#### Prediction Synthesizer → Explainer

When prediction needs explanation:

```python
# Prediction Synthesizer sets in response
next_agent = "explainer"
handoff_context = {
    "upstream_agent": "prediction_synthesizer",
    "prediction_summary": {
        "entity_id": "hcp_12345",
        "prediction_target": "trx",
        "point_estimate": 52.3,
        "confidence": 0.82
    },
    "feature_importance": {
        "patient_volume_decile": 0.32,
        "hcp_specialty": 0.28,
        "engagement_frequency": 0.21
    },
    "reason": "Generate stakeholder-friendly explanation of prediction"
}
```

#### Heterogeneous Optimizer → Resource Optimizer

When segment-specific effects inform allocation:

```python
# Heterogeneous Optimizer sets in response
next_agent = "resource_optimizer"
handoff_context = {
    "upstream_agent": "heterogeneous_optimizer",
    "segment_effects": {
        "specialty=Oncology": {"cate": 0.42, "size": 1250},
        "specialty=Primary Care": {"cate": 0.08, "size": 3400}
    },
    "optimal_policy": {
        "high_responders": ["specialty=Oncology", "decile=9-10"],
        "low_responders": ["specialty=Primary Care", "decile=1-2"]
    },
    "reason": "Use CATE estimates as response coefficients for allocation optimization"
}
```

#### Gap Analyzer → Resource Optimizer

When opportunities need resource allocation:

```python
# Gap Analyzer sets in response
next_agent = "resource_optimizer"
handoff_context = {
    "upstream_agent": "gap_analyzer",
    "opportunities": [
        {
            "segment": "specialty=Oncology",
            "gap_size": 500,
            "expected_roi": 3.2,
            "addressable_value": 4500000
        },
        {
            "segment": "region=Northeast",
            "gap_size": 350,
            "expected_roi": 2.8,
            "addressable_value": 2800000
        }
    ],
    "reason": "Optimize resource allocation to close identified gaps"
}
```

---

## 5. Validation Rules

### 5.1 Input Validation

**Prediction Synthesizer:**
```python
def validate_prediction_synthesizer_input(state: PredictionSynthesizerState) -> List[str]:
    """Validation for Prediction Synthesizer inputs"""
    errors = []

    # Required fields
    if not state.get("entity_id"):
        errors.append("entity_id is required")
    if not state.get("entity_type"):
        errors.append("entity_type is required")
    if not state.get("prediction_target"):
        errors.append("prediction_target is required")
    if not state.get("features"):
        errors.append("features dict is required")

    # Entity type validation
    valid_types = ["hcp", "territory", "region", "patient"]
    if state.get("entity_type") and state["entity_type"] not in valid_types:
        errors.append(f"entity_type must be one of {valid_types}")

    # Confidence level validation
    if state.get("confidence_level"):
        if not (0.5 <= state["confidence_level"] <= 0.99):
            errors.append("confidence_level must be between 0.5 and 0.99")

    # Ensemble method validation
    valid_methods = ["average", "weighted", "stacking", "voting"]
    if state.get("ensemble_method") and state["ensemble_method"] not in valid_methods:
        errors.append(f"ensemble_method must be one of {valid_methods}")

    return errors
```

**Resource Optimizer:**
```python
def validate_resource_optimizer_input(state: ResourceOptimizerState) -> List[str]:
    """Validation for Resource Optimizer inputs"""
    errors = []

    # Required fields
    if not state.get("allocation_targets") or len(state["allocation_targets"]) == 0:
        errors.append("At least one allocation target is required")
    if not state.get("constraints") or len(state["constraints"]) == 0:
        errors.append("At least one constraint is required")

    # Allocation targets validation
    for target in state.get("allocation_targets", []):
        if target.get("current_allocation", -1) < 0:
            errors.append(f"current_allocation must be >= 0 for {target.get('entity_id')}")
        if target.get("expected_response", 0) < 0:
            errors.append(f"expected_response must be >= 0 for {target.get('entity_id')}")

        min_alloc = target.get("min_allocation")
        max_alloc = target.get("max_allocation")
        if min_alloc is not None and max_alloc is not None:
            if min_alloc > max_alloc:
                errors.append(f"min_allocation > max_allocation for {target.get('entity_id')}")

    # Budget constraint check
    budget_constraints = [c for c in state.get("constraints", []) if c.get("constraint_type") == "budget"]
    if not budget_constraints:
        errors.append("At least one budget constraint is required")

    # Solver configuration
    if state.get("time_limit_seconds"):
        if not (1 <= state["time_limit_seconds"] <= 60):
            errors.append("time_limit_seconds must be between 1 and 60")
    if state.get("gap_tolerance"):
        if not (0 <= state["gap_tolerance"] <= 0.1):
            errors.append("gap_tolerance must be between 0 and 0.1")

    return errors
```

### 5.2 Output Validation

```python
def validate_tier4_output(output: Dict[str, Any], agent_name: str) -> List[str]:
    """Validate Tier 4 agent outputs"""
    errors = []

    # Common required fields
    required = ["total_latency_ms", "warnings", "timestamp"]
    for field in required:
        if field not in output:
            errors.append(f"{agent_name} output missing required field: {field}")

    # Latency sanity check
    max_latency = {
        "prediction_synthesizer": 15000,  # 15s
        "resource_optimizer": 20000       # 20s
    }
    if "total_latency_ms" in output:
        if output["total_latency_ms"] > max_latency.get(agent_name, 20000):
            errors.append(f"{agent_name} exceeded maximum latency")

    # Agent-specific validation
    if agent_name == "prediction_synthesizer":
        if "ensemble_prediction" in output:
            pred = output["ensemble_prediction"]
            if "confidence" in pred:
                if not (0.0 <= pred["confidence"] <= 1.0):
                    errors.append("ensemble_prediction.confidence must be between 0.0 and 1.0")
            if "model_agreement" in pred:
                if not (0.0 <= pred["model_agreement"] <= 1.0):
                    errors.append("ensemble_prediction.model_agreement must be between 0.0 and 1.0")

    elif agent_name == "resource_optimizer":
        if "solver_status" in output:
            if output["solver_status"] not in ["optimal", "infeasible", "failed"]:
                errors.append("solver_status must be 'optimal', 'infeasible', or 'failed'")
        if "projected_roi" in output:
            if output["projected_roi"] < 0:
                errors.append("projected_roi must be >= 0")

    return errors
```

---

## 6. Error Handling

### 6.1 Prediction Synthesizer Error Patterns

**All Models Failed:**
```python
if not predictions:
    return {
        **state,
        "errors": [{"node": "orchestrator", "error": "All models failed"}],
        "status": "failed"
    }
```

**Model Timeout:**
```python
try:
    result = await asyncio.wait_for(
        client.predict(...),
        timeout=5  # 5 second timeout per model
    )
except asyncio.TimeoutError:
    # Skip this model, continue with others
    state = {
        **state,
        "warnings": state.get("warnings", []) + [f"Model {model_id} timed out"]
    }
```

**Context Enrichment Failure (Non-Fatal):**
```python
# Context enrichment is optional - don't fail entire prediction
except Exception as e:
    return {
        **state,
        "warnings": state.get("warnings", []) + [f"Context enrichment failed: {str(e)}"],
        "status": "completed"  # Still completed
    }
```

### 6.2 Resource Optimizer Error Patterns

**Infeasible Problem:**
```python
if result["status"] == "infeasible":
    return {
        **state,
        "solver_status": "infeasible",
        "errors": [{"node": "optimizer", "error": "Problem is infeasible - constraints cannot be satisfied"}],
        "recommendations": ["Relax constraints or adjust allocation bounds"],
        "status": "failed"
    }
```

**Solver Timeout:**
```python
# Return best solution found so far
if not result.success and time_elapsed >= time_limit:
    return {
        **state,
        "solver_status": "timeout",
        "optimal_allocations": partial_solution,  # Best so far
        "warnings": [f"Solver timed out after {time_limit}s - returning best solution found"],
        "status": "projecting"  # Continue with partial solution
    }
```

**Invalid Constraints:**
```python
validation_errors = self._validate_inputs(targets, constraints)
if validation_errors:
    return {
        **state,
        "errors": [{"node": "formulator", "error": e} for e in validation_errors],
        "status": "failed"
    }
```

---

## 7. Performance Requirements

| Agent | Target Latency | Max Latency | Throughput | Model/Solver Calls |
|-------|----------------|-------------|------------|-------------------|
| **Prediction Synthesizer** | 8-10s | 15s | 4 queries/min | 3-5 model calls (parallel) |
| **Resource Optimizer** | 10-15s | 20s | 3 queries/min | 1 solver call |

### 7.1 Latency Breakdown

**Prediction Synthesizer:**
- Model orchestration (parallel): 5-8s
- Ensemble combination: <1s
- Context enrichment: 2-3s
- Total: 8-10s

**Resource Optimizer:**
- Problem formulation: 1-2s
- Optimization solve: 5-10s
- Scenario analysis (optional): 2-4s
- Impact projection: 1-2s
- Total: 10-15s

---

## 8. Testing Requirements

### 8.1 Unit Tests

**Prediction Synthesizer:**
- Model orchestration with timeouts
- Ensemble methods (average, weighted, voting)
- Confidence interval calculation
- Model agreement scoring
- Context enrichment

**Resource Optimizer:**
- Problem formulation
- Linear programming solver
- MILP solver
- Constraint validation
- Allocation result generation
- Impact projection

### 8.2 Integration Tests

```python
async def test_tier4_agent_integration(agent_name: str):
    """Test full agent execution"""

    # Setup
    state = create_test_input(agent_name)
    graph = build_agent_graph(agent_name)

    # Execute
    result = await graph.ainvoke(state)

    # Validate output contract
    errors = validate_tier4_output(result, agent_name)
    assert len(errors) == 0, f"Output validation failed: {errors}"

    # Check required fields
    assert result["status"] == "completed"
    assert result["total_latency_ms"] > 0
    assert result["timestamp"]

    # Agent-specific checks
    if agent_name == "prediction_synthesizer":
        assert result["ensemble_prediction"]
        assert result["models_succeeded"] > 0
    elif agent_name == "resource_optimizer":
        assert result["optimal_allocations"]
        assert result["solver_status"] == "optimal"
```

### 8.3 End-to-End Tests

```python
async def test_tier4_orchestration_flow():
    """Test orchestrator dispatching to Tier 4 agents"""

    # Prediction Synthesizer → Explainer flow
    query = "Predict TRx for HCP and explain the drivers"

    # Stage 1: Prediction Synthesizer
    pred_result = await orchestrator.dispatch("prediction_synthesizer", query, {
        "entity_id": "hcp_12345",
        "entity_type": "hcp",
        "prediction_target": "trx",
        "features": {...}
    })
    assert pred_result["status"] == "completed"
    assert pred_result["agent_result"]["ensemble_prediction"] is not None

    # Stage 2: Explainer (triggered by handoff)
    if pred_result["next_agent"] == "explainer":
        explain_result = await orchestrator.dispatch(
            "explainer",
            query,
            pred_result["handoff_context"]
        )
        assert explain_result["status"] == "completed"
```

---

## 9. Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-18 | Initial Tier 4 contracts | Claude |

---

## 10. Related Documents

- `base-contract.md` - Base agent structures
- `orchestrator-contracts.md` - Orchestrator communication
- `tier0-contracts.md` - ML Foundation contracts
- `tier2-contracts.md` - Causal Inference contracts
- `tier3-contracts.md` - Design & Monitoring contracts
- `agent-handoff.yaml` - Standard handoff format examples
- `.claude/specialists/Agent_Specialists_Tiers 1-5/prediction-synthesizer.md` - Prediction Synthesizer specialist
- `.claude/specialists/Agent_Specialists_Tiers 1-5/resource-optimizer.md` - Resource Optimizer specialist

---

## 11. DSPy Signal Contracts

Tier 4 agents have mixed DSPy roles (from E2I DSPy Feedback Learner Architecture V2):

| Agent | DSPy Role | Primary Signature | Behavior |
|-------|-----------|-------------------|----------|
| **Prediction Synthesizer** | Sender | `EvidenceSynthesisSignature` | Generates signals |
| **Resource Optimizer** | Recipient | N/A | Consumes optimized prompts |

### 11.1 Sender Agent (Prediction Synthesizer)

```python
class PredictionSynthesizerAgent(DSPySenderMixin):
    """
    Prediction Synthesizer is a Sender that generates training signals
    when synthesizing multi-model predictions.
    """

    @property
    def agent_name(self) -> str:
        return "prediction_synthesizer"

    @property
    def primary_signature(self) -> str:
        return "EvidenceSynthesisSignature"

    async def _synthesize_predictions(self, state: PredictionSynthesizerState) -> Dict:
        # ... multi-model synthesis logic ...
        result = await self._call_synthesis(model_outputs)

        # Collect training signal
        self.collect_training_signal(
            input_data={
                "model_predictions": model_outputs,
                "query": state["query"]
            },
            output_data={
                "synthesized_prediction": result["prediction"],
                "model_weights": result["weights"]
            },
            quality_score=result.get("quality", 0.8),
            signature_name="EvidenceSynthesisSignature",
            confidence=result.get("confidence", 0.8),
            session_id=state.get("session_id")
        )

        return result
```

### 11.2 Recipient Agent (Resource Optimizer)

```python
class ResourceOptimizerAgent(DSPyRecipientMixin):
    """
    Resource Optimizer is a Recipient that consumes optimized prompts
    for allocation recommendations.
    """

    async def _generate_allocation(self, state: ResourceOptimizerState) -> Dict:
        # Use optimized prompt if available
        prompt = self.get_optimized_prompt(
            "AllocationRecommendation",
            default=self._default_allocation_prompt
        )

        result = await self._call_llm(prompt, state)
        return result
```

### 11.3 Signal Quality Thresholds

```python
TIER4_SIGNAL_QUALITY_THRESHOLDS = {
    "prediction_synthesizer": {
        "min_quality": 0.6,
        "min_confidence": 0.5,
        "max_latency_ms": 45000,
        "required_fields": ["synthesized_prediction", "model_weights"]
    }
}
```
