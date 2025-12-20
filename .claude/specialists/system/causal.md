# Causal Engine Specialist Instructions

## Domain Scope
You are the Causal Engine specialist. Your scope is LIMITED to:
- `src/causal_engine/` - All causal inference modules
- `config/causal_config.yaml` - Causal inference settings

## Technology Stack
- **DoWhy**: Causal model definition and identification
- **EconML**: Heterogeneous treatment effect estimation
- **NetworkX**: DAG construction and path analysis

## Module Responsibilities

### dag_builder.py
Construct causal DAGs from domain knowledge:
```python
class CausalDAGBuilder:
    """
    Build DAGs for E2I causal relationships.
    
    Standard E2I Causal Structure:
    - Treatment: HCP engagement actions (triggers, calls, samples)
    - Outcome: Business KPIs (TRx, NRx, conversion)
    - Confounders: HCP characteristics, region, time
    """
    
    def build_treatment_outcome_dag(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> nx.DiGraph:
        pass
```

### effect_estimator.py
Estimate causal effects using DoWhy/EconML:
```python
class CausalEffectEstimator:
    """
    Estimate Average Treatment Effect (ATE) and 
    Conditional Average Treatment Effect (CATE).
    
    Methods:
    - Propensity Score Matching
    - Inverse Probability Weighting
    - Double Machine Learning (DML)
    - Causal Forests
    """
    
    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        method: str = "dml"
    ) -> EffectEstimate:
        pass
```

### path_analyzer.py
Trace causal paths through the DAG:
```python
class CausalPathAnalyzer:
    """
    Identify and analyze causal pathways.
    
    Use cases:
    - "Why did conversion drop?" → Find paths from confounders to outcome
    - "What drives TRx?" → Find all paths to TRx node
    """
    
    def find_causal_paths(
        self,
        dag: nx.DiGraph,
        source: str,
        target: str
    ) -> List[CausalPath]:
        pass
```

### counterfactual.py
What-if simulation engine:
```python
class CounterfactualSimulator:
    """
    Answer "what if" questions.
    
    Examples:
    - "What if we increased calls by 20%?"
    - "What if we hadn't triggered these HCPs?"
    """
    
    def simulate_intervention(
        self,
        model: CausalModel,
        intervention: Dict[str, float],
        target_outcome: str
    ) -> CounterfactualResult:
        pass
```

### heterogeneous_effects.py
CATE estimation by segment:
```python
class HeterogeneousEffectEstimator:
    """
    Estimate treatment effects by HCP segment.
    
    Segments:
    - By region
    - By specialty
    - By decile
    - By engagement history
    """
    
    def estimate_cate_by_segment(
        self,
        data: pd.DataFrame,
        segment_col: str,
        treatment: str,
        outcome: str
    ) -> Dict[str, EffectEstimate]:
        pass
```

### validators.py
Check causal assumptions:
```python
class CausalAssumptionValidator:
    """
    Validate causal inference assumptions:
    - Positivity (overlap)
    - Unconfoundedness (no unmeasured confounders)
    - SUTVA (no interference)
    """
    
    def check_positivity(self, propensity_scores: np.ndarray) -> ValidationResult:
        pass
    
    def check_balance(self, data: pd.DataFrame, treatment: str) -> ValidationResult:
        pass
```

### refutation.py
Robustness tests:
```python
class CausalRefutation:
    """
    DoWhy refutation tests:
    - Add random common cause
    - Placebo treatment
    - Data subset
    - Bootstrap
    """
    
    def run_all_refutations(self, model: CausalModel) -> RefutationReport:
        pass
```

## Pydantic Models (src/causal_engine/models/)

### causal_graph.py
```python
class CausalNode(BaseModel):
    name: str
    node_type: Literal["treatment", "outcome", "confounder", "mediator"]
    data_source: str  # Table/column reference

class CausalEdge(BaseModel):
    source: str
    target: str
    edge_type: Literal["causal", "confounding", "mediation"]

class CausalDAG(BaseModel):
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    assumptions: List[str]
```

### effect_models.py
```python
class EffectEstimate(BaseModel):
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    p_value: float
    method: str
    refutation_passed: bool
    sample_size: int
```

## Critical Rules

### 1. Always Validate Assumptions
```python
# Before any effect estimation:
validator = CausalAssumptionValidator()
positivity = validator.check_positivity(propensity_scores)
if not positivity.passed:
    raise CausalAssumptionViolation("Positivity violated: extreme propensity scores")
```

### 2. Always Run Refutations
```python
# After effect estimation:
refuter = CausalRefutation()
report = refuter.run_all_refutations(model)
if not report.all_passed:
    estimate.confidence = "low"
    estimate.warnings = report.failed_tests
```

### 3. Document Assumptions
Every causal model must document:
- Treatment definition
- Outcome definition
- Assumed confounders (and why)
- Excluded confounders (and why)

## Integration Contracts

### Input Contract (from Agents)
```python
class CausalQuery(BaseModel):
    query_type: Literal["ate", "cate", "path", "counterfactual"]
    treatment: str
    outcome: str
    confounders: List[str]
    data_filter: Optional[Dict]
    segment_by: Optional[str]
```

### Output Contract (to Agents)
```python
class CausalResult(BaseModel):
    query_type: str
    estimate: Optional[EffectEstimate]
    paths: Optional[List[CausalPath]]
    counterfactual: Optional[CounterfactualResult]
    assumptions_validated: bool
    refutation_report: RefutationReport
```

## Testing Requirements
- `tests/unit/test_causal_engine/`
- All effect estimates must include CI and p-value
- Refutation tests must pass for production use

## Handoff Format
```yaml
causal_handoff:
  result_type: <ate|cate|path|counterfactual>
  estimate:
    value: <float>
    confidence_interval: [lower, upper]
    p_value: <float>
  assumptions_valid: <bool>
  refutation_passed: <bool>
  interpretation: <natural language summary>
```
