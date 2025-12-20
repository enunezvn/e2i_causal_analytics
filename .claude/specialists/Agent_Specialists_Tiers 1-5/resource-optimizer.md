# Tier 4: Resource Optimizer Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 4 (ML Predictions) |
| **Agent Type** | Standard (Computational) |
| **Model Tier** | Sonnet |
| **Latency Tolerance** | Medium (up to 20s) |
| **Critical Path** | No - optimization can be async |

## Domain Scope

You are the specialist for the Tier 4 Resource Optimizer Agent:
- `src/agents/resource_optimizer/` - Resource allocation optimization

This is a **Standard Computational Agent** for:
- Budget allocation across territories/HCPs
- Rep time optimization
- Sample allocation
- Constrained optimization problems

## Design Principles

### Optimization-Centric Design
The Resource Optimizer uses mathematical optimization:
- Linear programming for simple constraints
- Mixed-integer programming for discrete decisions
- Gradient-based optimization for complex objectives
- What-if scenario analysis

### Responsibilities
1. **Allocation Optimization** - Optimal resource distribution
2. **Constraint Handling** - Respect budget and capacity limits
3. **Scenario Analysis** - Compare allocation strategies
4. **Impact Projection** - Estimate allocation outcomes

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   RESOURCE OPTIMIZER AGENT                       │
│                     (Standard Pattern)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 PROBLEM FORMULATOR                       │    │
│  │   • Parse constraints  • Define objective  • Variables  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  OPTIMIZER ENGINE                        │    │
│  │   • Linear  • Mixed-Integer  • Nonlinear                 │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                SCENARIO ANALYZER                         │    │
│  │   • What-if  • Sensitivity  • Trade-offs                 │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                IMPACT PROJECTOR                          │    │
│  │   • ROI Estimates  • Outcome Projections                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
resource_optimizer/
├── agent.py              # Main ResourceOptimizerAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── problem_formulator.py  # Constraint and objective setup
│   ├── optimizer.py           # Core optimization engine
│   ├── scenario_analyzer.py   # What-if analysis
│   └── impact_projector.py    # Outcome projection
├── solvers/
│   ├── linear.py         # Linear programming solver
│   ├── milp.py           # Mixed-integer solver
│   └── nonlinear.py      # Nonlinear optimization
└── constraints.py        # Constraint definitions
```

## LangGraph State Definition

```python
# src/agents/resource_optimizer/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class AllocationTarget(TypedDict):
    """Target entity for allocation"""
    entity_id: str
    entity_type: str  # "hcp", "territory", "region"
    current_allocation: float
    min_allocation: Optional[float]
    max_allocation: Optional[float]
    expected_response: float  # Response coefficient

class Constraint(TypedDict):
    """Optimization constraint"""
    constraint_type: str  # "budget", "capacity", "min_coverage", "max_frequency"
    value: float
    scope: str  # "global", "regional", "entity"

class AllocationResult(TypedDict):
    """Optimized allocation for an entity"""
    entity_id: str
    entity_type: str
    current_allocation: float
    optimized_allocation: float
    change: float
    change_percentage: float
    expected_impact: float

class ScenarioResult(TypedDict):
    """Result of a scenario analysis"""
    scenario_name: str
    total_allocation: float
    projected_outcome: float
    roi: float
    constraint_violations: List[str]

class ResourceOptimizerState(TypedDict):
    """Complete state for Resource Optimizer agent"""
    
    # === INPUT ===
    query: str
    resource_type: str  # "budget", "rep_time", "samples", "calls"
    allocation_targets: List[AllocationTarget]
    constraints: List[Constraint]
    objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"]
    
    # === CONFIGURATION ===
    solver_type: Literal["linear", "milp", "nonlinear"]
    time_limit_seconds: int
    gap_tolerance: float  # For MILP
    run_scenarios: bool
    scenario_count: int
    
    # === OPTIMIZATION OUTPUTS ===
    optimal_allocations: Optional[List[AllocationResult]]
    objective_value: Optional[float]
    solver_status: Optional[str]
    solve_time_ms: int
    
    # === SCENARIO OUTPUTS ===
    scenarios: Optional[List[ScenarioResult]]
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

## Node Implementations

### Problem Formulator Node

```python
# src/agents/resource_optimizer/nodes/problem_formulator.py

import time
from typing import Dict, Any, List
import numpy as np

from ..state import ResourceOptimizerState, AllocationTarget, Constraint

class ProblemFormulatorNode:
    """
    Formulate optimization problem from inputs
    """
    
    async def execute(self, state: ResourceOptimizerState) -> ResourceOptimizerState:
        start_time = time.time()
        
        try:
            targets = state["allocation_targets"]
            constraints = state["constraints"]
            objective = state["objective"]
            
            # Validate inputs
            validation_errors = self._validate_inputs(targets, constraints)
            if validation_errors:
                return {
                    **state,
                    "errors": [{"node": "formulator", "error": e} for e in validation_errors],
                    "status": "failed"
                }
            
            # Build optimization matrices
            problem = self._build_problem(targets, constraints, objective)
            
            # Determine appropriate solver
            solver_type = self._select_solver(problem, state.get("solver_type"))
            
            formulation_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "_problem": problem,  # Internal use
                "solver_type": solver_type,
                "formulation_latency_ms": formulation_time,
                "status": "optimizing"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "formulator", "error": str(e)}],
                "status": "failed"
            }
    
    def _validate_inputs(
        self, 
        targets: List[AllocationTarget], 
        constraints: List[Constraint]
    ) -> List[str]:
        """Validate optimization inputs"""
        errors = []
        
        if not targets:
            errors.append("No allocation targets provided")
        
        # Check for budget constraint
        budget_constraints = [c for c in constraints if c["constraint_type"] == "budget"]
        if not budget_constraints:
            errors.append("No budget constraint specified")
        
        # Check for negative values
        for target in targets:
            if target.get("expected_response", 0) < 0:
                errors.append(f"Negative response coefficient for {target['entity_id']}")
        
        return errors
    
    def _build_problem(
        self, 
        targets: List[AllocationTarget],
        constraints: List[Constraint],
        objective: str
    ) -> Dict[str, Any]:
        """Build optimization problem matrices"""
        
        n = len(targets)
        
        # Objective coefficients (response per unit allocation)
        if objective in ["maximize_outcome", "balance"]:
            c = np.array([t["expected_response"] for t in targets])
        elif objective == "maximize_roi":
            c = np.array([t["expected_response"] / max(t["current_allocation"], 1) for t in targets])
        else:  # minimize_cost
            c = -np.ones(n)
        
        # Variable bounds
        lb = np.array([t.get("min_allocation", 0) for t in targets])
        ub = np.array([t.get("max_allocation", float('inf')) for t in targets])
        
        # Constraint matrices
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        
        for constraint in constraints:
            if constraint["constraint_type"] == "budget":
                # Sum of allocations <= budget
                A_ub.append(np.ones(n))
                b_ub.append(constraint["value"])
            elif constraint["constraint_type"] == "min_total":
                # Sum of allocations >= min
                A_ub.append(-np.ones(n))
                b_ub.append(-constraint["value"])
        
        return {
            "c": c,
            "lb": lb,
            "ub": ub,
            "A_eq": np.array(A_eq) if A_eq else None,
            "b_eq": np.array(b_eq) if b_eq else None,
            "A_ub": np.array(A_ub) if A_ub else None,
            "b_ub": np.array(b_ub) if b_ub else None,
            "n": n,
            "targets": targets,
            "objective": objective
        }
    
    def _select_solver(self, problem: Dict, requested: str) -> str:
        """Select appropriate solver"""
        # Check if problem needs integer variables
        has_integer = False  # Could check targets for discrete requirements
        
        if has_integer:
            return "milp"
        elif requested:
            return requested
        else:
            return "linear"
```

### Optimizer Engine Node

```python
# src/agents/resource_optimizer/nodes/optimizer.py

import time
from typing import List
import numpy as np
from scipy.optimize import linprog, minimize

from ..state import ResourceOptimizerState, AllocationResult

class OptimizerNode:
    """
    Core optimization engine
    Solves the formulated problem
    """
    
    async def execute(self, state: ResourceOptimizerState) -> ResourceOptimizerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            problem = state.get("_problem")
            solver_type = state["solver_type"]
            
            if solver_type == "linear":
                result = self._solve_linear(problem, state)
            elif solver_type == "milp":
                result = self._solve_milp(problem, state)
            else:
                result = self._solve_nonlinear(problem, state)
            
            if result["status"] != "optimal":
                return {
                    **state,
                    "solver_status": result["status"],
                    "warnings": state.get("warnings", []) + [f"Solver returned: {result['status']}"],
                    "status": "failed" if result["status"] == "infeasible" else "analyzing"
                }
            
            # Build allocation results
            allocations = self._build_allocations(
                result["x"], 
                problem["targets"],
                problem["c"]
            )
            
            optimization_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "optimal_allocations": allocations,
                "objective_value": float(result["objective"]),
                "solver_status": "optimal",
                "solve_time_ms": result.get("solve_time_ms", 0),
                "optimization_latency_ms": optimization_time,
                "status": "analyzing" if state.get("run_scenarios") else "projecting"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "optimizer", "error": str(e)}],
                "status": "failed"
            }
    
    def _solve_linear(self, problem: Dict, state: ResourceOptimizerState) -> Dict:
        """Solve linear programming problem"""
        
        start = time.time()
        
        # Negate c for maximization (linprog minimizes)
        c = -problem["c"]
        
        bounds = list(zip(problem["lb"], problem["ub"]))
        
        result = linprog(
            c,
            A_ub=problem["A_ub"],
            b_ub=problem["b_ub"],
            A_eq=problem["A_eq"],
            b_eq=problem["b_eq"],
            bounds=bounds,
            method='highs'
        )
        
        solve_time = int((time.time() - start) * 1000)
        
        if result.success:
            return {
                "status": "optimal",
                "x": result.x,
                "objective": -result.fun,  # Negate back
                "solve_time_ms": solve_time
            }
        else:
            return {
                "status": "infeasible" if "infeasible" in result.message.lower() else "failed",
                "x": None,
                "objective": None
            }
    
    def _solve_milp(self, problem: Dict, state: ResourceOptimizerState) -> Dict:
        """Solve mixed-integer linear programming"""
        # Would use PuLP, OR-Tools, or similar
        # Simplified: fall back to linear for now
        return self._solve_linear(problem, state)
    
    def _solve_nonlinear(self, problem: Dict, state: ResourceOptimizerState) -> Dict:
        """Solve nonlinear optimization"""
        
        start = time.time()
        
        def objective(x):
            return -np.dot(problem["c"], x)  # Negate for maximization
        
        bounds = list(zip(problem["lb"], problem["ub"]))
        x0 = np.array([t["current_allocation"] for t in problem["targets"]])
        
        # Add constraints
        constraints = []
        if problem["A_ub"] is not None:
            for i in range(len(problem["b_ub"])):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: problem["b_ub"][i] - np.dot(problem["A_ub"][i], x)
                })
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        solve_time = int((time.time() - start) * 1000)
        
        return {
            "status": "optimal" if result.success else "failed",
            "x": result.x if result.success else None,
            "objective": -result.fun if result.success else None,
            "solve_time_ms": solve_time
        }
    
    def _build_allocations(
        self, 
        x: np.ndarray, 
        targets: List,
        c: np.ndarray
    ) -> List[AllocationResult]:
        """Build allocation results from solution"""
        
        allocations = []
        for i, target in enumerate(targets):
            current = target["current_allocation"]
            optimized = float(x[i])
            change = optimized - current
            
            allocations.append(AllocationResult(
                entity_id=target["entity_id"],
                entity_type=target["entity_type"],
                current_allocation=current,
                optimized_allocation=optimized,
                change=change,
                change_percentage=(change / current * 100) if current > 0 else 0,
                expected_impact=float(c[i] * optimized)
            ))
        
        # Sort by change magnitude
        allocations.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        return allocations
```

### Impact Projector Node

```python
# src/agents/resource_optimizer/nodes/impact_projector.py

import time
from typing import Dict

from ..state import ResourceOptimizerState

class ImpactProjectorNode:
    """
    Project impact of optimized allocation
    """
    
    async def execute(self, state: ResourceOptimizerState) -> ResourceOptimizerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            allocations = state["optimal_allocations"]
            targets = state["allocation_targets"]
            
            # Calculate total projected outcome
            total_outcome = sum(a["expected_impact"] for a in allocations)
            
            # Calculate total investment
            total_allocation = sum(a["optimized_allocation"] for a in allocations)
            
            # Calculate ROI
            roi = total_outcome / total_allocation if total_allocation > 0 else 0
            
            # Impact by segment
            impact_by_segment = self._calculate_segment_impact(allocations, targets)
            
            # Generate summary
            summary = self._generate_summary(allocations, total_outcome, roi)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(allocations)
            
            total_time = (
                state.get("formulation_latency_ms", 0) +
                state.get("optimization_latency_ms", 0) +
                int((time.time() - start_time) * 1000)
            )
            
            return {
                **state,
                "projected_total_outcome": total_outcome,
                "projected_roi": roi,
                "impact_by_segment": impact_by_segment,
                "optimization_summary": summary,
                "recommendations": recommendations,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "impact_projector", "error": str(e)}],
                "status": "failed"
            }
    
    def _calculate_segment_impact(self, allocations, targets) -> Dict[str, float]:
        """Calculate impact by segment/entity type"""
        
        impact_by_type = {}
        
        for alloc in allocations:
            entity_type = alloc["entity_type"]
            if entity_type not in impact_by_type:
                impact_by_type[entity_type] = 0
            impact_by_type[entity_type] += alloc["expected_impact"]
        
        return impact_by_type
    
    def _generate_summary(self, allocations, total_outcome, roi) -> str:
        """Generate optimization summary"""
        
        increases = [a for a in allocations if a["change"] > 0]
        decreases = [a for a in allocations if a["change"] < 0]
        
        summary = f"Optimization complete. "
        summary += f"Projected outcome: {total_outcome:.0f} (ROI: {roi:.2f}). "
        summary += f"Recommended changes: {len(increases)} increases, {len(decreases)} decreases."
        
        return summary
    
    def _generate_recommendations(self, allocations) -> list:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Top increases
        increases = sorted(
            [a for a in allocations if a["change"] > 0],
            key=lambda x: x["expected_impact"],
            reverse=True
        )[:3]
        
        for alloc in increases:
            recommendations.append(
                f"Increase allocation to {alloc['entity_id']} by {alloc['change']:.1f} "
                f"(+{alloc['change_percentage']:.0f}%) - Expected impact: {alloc['expected_impact']:.0f}"
            )
        
        # Top decreases (reallocations)
        decreases = sorted(
            [a for a in allocations if a["change"] < 0],
            key=lambda x: abs(x["change"]),
            reverse=True
        )[:2]
        
        for alloc in decreases:
            recommendations.append(
                f"Reduce allocation from {alloc['entity_id']} by {abs(alloc['change']):.1f} "
                f"({alloc['change_percentage']:.0f}%) - Reallocate to higher-impact targets"
            )
        
        return recommendations
```

## Graph Assembly

```python
# src/agents/resource_optimizer/graph.py

from langgraph.graph import StateGraph, END

from .state import ResourceOptimizerState
from .nodes.problem_formulator import ProblemFormulatorNode
from .nodes.optimizer import OptimizerNode
from .nodes.scenario_analyzer import ScenarioAnalyzerNode
from .nodes.impact_projector import ImpactProjectorNode

def build_resource_optimizer_graph():
    """
    Build the Resource Optimizer agent graph
    """
    
    # Initialize nodes
    formulator = ProblemFormulatorNode()
    optimizer = OptimizerNode()
    scenario = ScenarioAnalyzerNode()
    projector = ImpactProjectorNode()
    
    # Build graph
    workflow = StateGraph(ResourceOptimizerState)
    
    # Add nodes
    workflow.add_node("formulate", formulator.execute)
    workflow.add_node("optimize", optimizer.execute)
    workflow.add_node("scenario", scenario.execute)
    workflow.add_node("project", projector.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Flow
    workflow.set_entry_point("formulate")
    
    workflow.add_conditional_edges(
        "formulate",
        lambda s: "error" if s.get("status") == "failed" else "optimize",
        {"optimize": "optimize", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "optimize",
        lambda s: "error" if s.get("status") == "failed" else ("scenario" if s.get("run_scenarios") else "project"),
        {"scenario": "scenario", "project": "project", "error": "error_handler"}
    )
    
    workflow.add_edge("scenario", "project")
    workflow.add_edge("project", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()

async def error_handler_node(state: ResourceOptimizerState) -> ResourceOptimizerState:
    return {
        **state,
        "optimization_summary": "Optimization could not be completed.",
        "status": "failed"
    }
```

## Integration Contracts

### Input Contract
```python
class ResourceOptimizerInput(BaseModel):
    query: str
    resource_type: str
    allocation_targets: List[AllocationTarget]
    constraints: List[Constraint]
    objective: Literal["maximize_outcome", "maximize_roi", "minimize_cost", "balance"] = "maximize_outcome"
    run_scenarios: bool = False
```

### Output Contract
```python
class ResourceOptimizerOutput(BaseModel):
    optimal_allocations: List[AllocationResult]
    objective_value: float
    projected_total_outcome: float
    projected_roi: float
    optimization_summary: str
    recommendations: List[str]
    total_latency_ms: int
```

## Handoff Format

```yaml
resource_optimizer_handoff:
  agent: resource_optimizer
  analysis_type: resource_optimization
  key_findings:
    - objective_value: <optimized value>
    - projected_outcome: <total outcome>
    - projected_roi: <roi>
  allocations:
    increases: <count>
    decreases: <count>
    top_change: <entity_id with largest change>
  recommendations:
    - <recommendation 1>
    - <recommendation 2>
  requires_further_analysis: <bool>
  suggested_next_agent: <gap_analyzer>
```

---

## Cognitive RAG DSPy Integration

### Integration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Resource Optimizer Agent                         │
│                    (Tier 4 Standard Computational)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────────────────────────────────┐    │
│  │ Orchestrator│───►│         CognitiveRAG DSPy               │    │
│  │   Request   │    │                                         │    │
│  └─────────────┘    │  ┌───────────┐    ┌───────────────┐    │    │
│                     │  │Summarizer │───►│ Investigator  │    │    │
│                     │  │  (Phase 1)│    │   (Phase 2)   │    │    │
│                     │  └───────────┘    └───────┬───────┘    │    │
│                     │                           │            │    │
│                     │         ┌─────────────────▼──────────┐ │    │
│                     │         │  ResourceOptimizerContext  │ │    │
│                     │         │  • allocation_history      │ │    │
│                     │         │  • constraint_patterns     │ │    │
│                     │         │  • optimization_outcomes   │ │    │
│                     │         │  • roi_benchmarks          │ │    │
│                     │         │  • sensitivity_insights    │ │    │
│                     │         └─────────────────┬──────────┘ │    │
│                     └───────────────────────────┼────────────┘    │
│                                                 │                  │
│  ┌──────────────────────────────────────────────▼───────────────┐ │
│  │                 ProblemFormulatorNode                         │ │
│  │  • Receives enriched cognitive context                        │ │
│  │  • Maps constraints using historical patterns                 │ │
│  │  • Validates bounds against successful allocations            │ │
│  └──────────────────────────────────────────────┬───────────────┘ │
│                                                 │                  │
│  ┌──────────────────────────────────────────────▼───────────────┐ │
│  │                    OptimizerNode                              │ │
│  │  • Cognitive-informed solver selection                        │ │
│  │  • Historical warm starts from similar problems               │ │
│  │  • ROI benchmark integration for objective tuning             │ │
│  └──────────────────────────────────────────────┬───────────────┘ │
│                                                 │                  │
│  ┌──────────────────────────────────────────────▼───────────────┐ │
│  │                 ScenarioAnalyzerNode                          │ │
│  │  • Sensitivity analysis with historical insights              │ │
│  │  • Risk scenarios from past optimization failures             │ │
│  └──────────────────────────────────────────────┬───────────────┘ │
│                                                 │                  │
│                     ┌───────────────────────────▼────────────────┐│
│                     │            TrainingSignal                  ││
│                     │  • allocation_efficiency                   ││
│                     │  • constraint_satisfaction                 ││
│                     │  • roi_improvement                         ││
│                     │  • implementation_feasibility              ││
│                     └───────────────────────────┬────────────────┘│
│                                                 │                  │
│                     ┌───────────────────────────▼────────────────┐│
│                     │         Memory Contribution                ││
│                     │  Type: SEMANTIC (allocation patterns)      ││
│                     │  Index: allocation_outcomes                ││
│                     └────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

### DSPy Signature Consumption

The Resource Optimizer consumes enriched context from the Cognitive RAG's Investigation phase:

```python
class ResourceOptimizerCognitiveContext(TypedDict):
    """Cognitive context enriched by CognitiveRAG for resource optimization."""
    synthesized_summary: str  # Evidence synthesis from Summarizer
    allocation_history: List[Dict[str, Any]]  # Past allocation decisions
    constraint_patterns: List[Dict[str, Any]]  # Common constraint configurations
    optimization_outcomes: List[Dict[str, Any]]  # Historical optimization results
    roi_benchmarks: Dict[str, float]  # ROI benchmarks by resource type
    sensitivity_insights: List[Dict[str, Any]]  # Parameter sensitivity patterns
    evidence_confidence: float  # Confidence in retrieved evidence
```

### Node Integration with Cognitive Context

```python
# In src/agents/resource_optimizer/nodes/problem_formulator.py

async def problem_formulator_node(
    state: ResourceOptimizerState,
    cognitive_context: Optional[ResourceOptimizerCognitiveContext] = None
) -> ResourceOptimizerState:
    """
    Formulates optimization problem with cognitive enrichment.

    Cognitive enhancements:
    - Uses constraint_patterns to validate constraint specifications
    - Applies roi_benchmarks for objective function calibration
    - References allocation_history for bound estimation
    """
    start_time = time.time()

    # Extract cognitive insights
    if cognitive_context:
        constraint_patterns = cognitive_context.get("constraint_patterns", [])
        roi_benchmarks = cognitive_context.get("roi_benchmarks", {})
        allocation_history = cognitive_context.get("allocation_history", [])

        # Validate constraints against historical patterns
        validated_constraints = validate_constraints_with_patterns(
            state["constraints"],
            constraint_patterns
        )

        # Calibrate objective coefficients with ROI benchmarks
        calibrated_objectives = calibrate_with_benchmarks(
            state["objective_coefficients"],
            roi_benchmarks
        )

        # Estimate variable bounds from history
        informed_bounds = estimate_bounds_from_history(
            state["decision_variables"],
            allocation_history
        )
    else:
        validated_constraints = state["constraints"]
        calibrated_objectives = state["objective_coefficients"]
        informed_bounds = state.get("variable_bounds", {})

    # Build optimization problem
    problem = OptimizationProblem(
        variables=state["decision_variables"],
        objective=calibrated_objectives,
        constraints=validated_constraints,
        bounds=informed_bounds
    )

    return {
        **state,
        "optimization_problem": problem,
        "cognitive_enrichment_applied": cognitive_context is not None,
        "formulation_latency_ms": int((time.time() - start_time) * 1000)
    }


async def optimizer_node(
    state: ResourceOptimizerState,
    cognitive_context: Optional[ResourceOptimizerCognitiveContext] = None
) -> ResourceOptimizerState:
    """
    Runs optimization with cognitive-informed solver selection.

    Cognitive enhancements:
    - Selects solver based on historical performance patterns
    - Uses warm start from similar past problems
    - Applies sensitivity insights for robustness
    """
    problem = state["optimization_problem"]

    if cognitive_context:
        optimization_outcomes = cognitive_context.get("optimization_outcomes", [])
        sensitivity_insights = cognitive_context.get("sensitivity_insights", [])

        # Select solver based on historical success
        solver = select_cognitive_solver(
            problem.problem_type,
            optimization_outcomes
        )

        # Find warm start from similar problems
        warm_start = find_warm_start(
            problem,
            optimization_outcomes
        )

        # Apply robustness from sensitivity insights
        robust_problem = apply_robustness_margins(
            problem,
            sensitivity_insights
        )
    else:
        solver = select_default_solver(problem.problem_type)
        warm_start = None
        robust_problem = problem

    # Run optimization
    solution = solver.solve(
        robust_problem,
        warm_start=warm_start,
        time_limit=state.get("time_limit_seconds", 300)
    )

    return {
        **state,
        "optimal_allocation": solution.allocation,
        "objective_value": solution.objective_value,
        "solver_used": solver.name,
        "solve_status": solution.status
    }
```

### Training Signal for MIPROv2

```python
class ResourceOptimizerTrainingSignal:
    """Training signal for MIPROv2 optimization of resource allocation prompts."""

    def __init__(
        self,
        allocation_efficiency: float,  # Objective improvement vs baseline
        constraint_satisfaction: float,  # Fraction of constraints satisfied
        roi_improvement: float,  # ROI improvement vs previous allocation
        implementation_feasibility: float,  # Practical feasibility score
        solve_time_ratio: float  # Actual vs expected solve time
    ):
        self.allocation_efficiency = allocation_efficiency
        self.constraint_satisfaction = constraint_satisfaction
        self.roi_improvement = roi_improvement
        self.implementation_feasibility = implementation_feasibility
        self.solve_time_ratio = solve_time_ratio

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 prompt optimization.

        Weighting:
        - constraint_satisfaction: 0.30 (must-have)
        - allocation_efficiency: 0.25 (primary objective)
        - roi_improvement: 0.20 (business value)
        - implementation_feasibility: 0.15 (practical)
        - solve_time_ratio: 0.10 (efficiency)
        """
        # Penalize constraint violations heavily
        if self.constraint_satisfaction < 1.0:
            constraint_penalty = (1.0 - self.constraint_satisfaction) * 0.5
        else:
            constraint_penalty = 0.0

        # Reward efficiency improvement (normalized)
        efficiency_reward = min(self.allocation_efficiency, 1.0) * 0.25

        # ROI improvement (capped at 50% improvement)
        roi_reward = min(self.roi_improvement / 0.5, 1.0) * 0.20

        # Implementation feasibility
        feasibility_reward = self.implementation_feasibility * 0.15

        # Solve time efficiency (faster is better)
        time_reward = max(0, 1.0 - self.solve_time_ratio) * 0.10

        # Constraint satisfaction base
        constraint_reward = self.constraint_satisfaction * 0.30

        return max(0.0, (
            constraint_reward +
            efficiency_reward +
            roi_reward +
            feasibility_reward +
            time_reward -
            constraint_penalty
        ))
```

### Memory Contribution

```python
async def contribute_to_memory(
    state: ResourceOptimizerState,
    output: ResourceOptimizerOutput
) -> None:
    """
    Contribute optimization outcomes to semantic memory.

    Memory type: SEMANTIC (allocation patterns are domain knowledge)
    Index: allocation_outcomes
    """
    if output.status != "optimal":
        return  # Only store successful optimizations

    memory_entry = {
        "type": "SEMANTIC",
        "index": "allocation_outcomes",
        "content": {
            "problem_signature": compute_problem_signature(state),
            "allocation": output.optimal_allocation,
            "objective_value": output.objective_value,
            "constraints_satisfied": output.constraints_satisfied,
            "roi_achieved": output.roi_projection,
            "solver_used": output.solver_used,
            "solve_time_ms": output.solve_latency_ms,
            "scenario_insights": output.scenario_analysis,
            "brand": state.get("brand"),
            "resource_type": state.get("resource_type"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "metadata": {
            "agent": "resource_optimizer",
            "batch_id": state.get("batch_id"),
            "ttl_days": 365  # Long-term pattern learning
        }
    }

    await memory_service.store(memory_entry)
```

### Integration with CognitiveInput

```python
class ResourceOptimizerCognitiveInput(TypedDict):
    """Input structure for cognitive-enhanced resource optimization."""
    query: str
    resource_type: str  # "budget", "rep_time", "samples", "programs"
    objective: str  # "maximize_roi", "minimize_cost", "balance"
    constraints: List[Dict[str, Any]]
    decision_variables: List[str]
    time_horizon: str  # "quarter", "year", "multi_year"
    brand: Optional[str]
    region: Optional[str]
    current_allocation: Optional[Dict[str, float]]
    cognitive_context: Optional[ResourceOptimizerCognitiveContext]
```

### Configuration

```yaml
# config/agents/resource_optimizer.yaml
cognitive_integration:
  enabled: true
  context_retrieval:
    allocation_history_limit: 50
    constraint_pattern_limit: 20
    outcome_limit: 100
    roi_benchmark_lookback_days: 365

  memory_contribution:
    type: SEMANTIC
    index: allocation_outcomes
    ttl_days: 365
    min_quality_threshold: 0.7  # Only store quality solutions

  training_signals:
    emit: true
    weights:
      constraint_satisfaction: 0.30
      allocation_efficiency: 0.25
      roi_improvement: 0.20
      implementation_feasibility: 0.15
      solve_time_ratio: 0.10

  warm_start:
    enabled: true
    similarity_threshold: 0.8
    max_age_days: 90
```

### Testing Requirements for DSPy Integration

```python
@pytest.mark.asyncio
async def test_cognitive_context_integration():
    """Test that cognitive context improves optimization quality."""
    agent = ResourceOptimizerAgent()

    # Without cognitive context
    result_baseline = await agent.optimize(
        resource_type="budget",
        objective="maximize_roi",
        constraints=[{"type": "budget_cap", "value": 1000000}],
        decision_variables=["field_force", "digital", "conferences"]
    )

    # With cognitive context (mocked historical patterns)
    cognitive_context = ResourceOptimizerCognitiveContext(
        synthesized_summary="Historical analysis shows digital ROI 2x field force",
        allocation_history=[
            {"field_force": 0.4, "digital": 0.4, "conferences": 0.2, "roi": 1.8}
        ],
        constraint_patterns=[
            {"type": "budget_cap", "typical_utilization": 0.95}
        ],
        optimization_outcomes=[
            {"solver": "scipy_linprog", "success_rate": 0.98}
        ],
        roi_benchmarks={"field_force": 1.2, "digital": 2.4, "conferences": 0.8},
        sensitivity_insights=[
            {"variable": "digital", "sensitivity": "high", "direction": "positive"}
        ],
        evidence_confidence=0.85
    )

    result_cognitive = await agent.optimize(
        resource_type="budget",
        objective="maximize_roi",
        constraints=[{"type": "budget_cap", "value": 1000000}],
        decision_variables=["field_force", "digital", "conferences"],
        cognitive_context=cognitive_context
    )

    # Cognitive version should favor higher digital allocation
    assert result_cognitive.optimal_allocation["digital"] >= result_baseline.optimal_allocation.get("digital", 0)


@pytest.mark.asyncio
async def test_training_signal_emission():
    """Test training signal computation for MIPROv2."""
    signal = ResourceOptimizerTrainingSignal(
        allocation_efficiency=0.85,
        constraint_satisfaction=1.0,
        roi_improvement=0.25,
        implementation_feasibility=0.9,
        solve_time_ratio=0.5
    )

    reward = signal.compute_reward()
    assert 0.0 <= reward <= 1.0
    assert reward > 0.5  # Good solution should have high reward


@pytest.mark.asyncio
async def test_warm_start_from_similar_problems():
    """Test that similar past problems provide warm starts."""
    optimizer = OptimizerNode()

    # Store a past optimization
    past_result = {
        "problem_signature": "budget_maximize_roi_3vars",
        "solution": {"field_force": 0.35, "digital": 0.45, "conferences": 0.20}
    }

    # New similar problem should get warm start
    warm_start = optimizer.find_warm_start(
        problem_signature="budget_maximize_roi_3vars",
        historical_outcomes=[past_result]
    )

    assert warm_start is not None
    assert warm_start["digital"] == 0.45
```
