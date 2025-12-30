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

### discovery/ (V4.4+)

Structure learning module for automatic DAG discovery from data:

```
src/causal_engine/discovery/
├── __init__.py                    # Exports DiscoveryRunner, DiscoveryGate, DriverRanker
├── base.py                        # Enums, dataclasses, protocols
├── runner.py                      # DiscoveryRunner - ensemble orchestration
├── gate.py                        # DiscoveryGate - quality gating decisions
├── driver_ranker.py               # DriverRanker - causal vs predictive comparison
└── algorithms/
    ├── __init__.py
    ├── base.py                    # DiscoveryAlgorithm protocol
    ├── ges_wrapper.py             # GES (Greedy Equivalence Search)
    └── pc_wrapper.py              # PC (Peter-Clark)
```

#### DiscoveryRunner
Orchestrates multi-algorithm ensemble structure learning:
```python
class DiscoveryRunner:
    """
    Run causal discovery algorithms in ensemble.

    Supported algorithms:
    - GES (Greedy Equivalence Search) - Score-based
    - PC (Peter-Clark) - Constraint-based

    Features:
    - Parallel algorithm execution
    - Edge voting across algorithms
    - Cycle removal for DAG enforcement
    """

    async def discover_dag(
        self,
        data: pd.DataFrame,
        config: Optional[DiscoveryConfig] = None,
        session_id: Optional[UUID] = None,
    ) -> DiscoveryResult:
        """
        Discover DAG structure from data.

        Args:
            data: DataFrame with feature columns
            config: DiscoveryConfig with algorithms, threshold, alpha
            session_id: Optional session for tracing

        Returns:
            DiscoveryResult with ensemble_dag, edges, algorithm_results
        """
        pass

    def discover_dag_sync(self, data: pd.DataFrame, config: Optional[DiscoveryConfig] = None) -> DiscoveryResult:
        """Synchronous wrapper for discover_dag."""
        pass
```

#### DiscoveryGate
Evaluates discovery quality and makes gate decisions:
```python
class DiscoveryGate:
    """
    Evaluate discovery quality and determine usage.

    Gate Decisions:
    - ACCEPT (>= 0.8): Use discovered DAG directly
    - AUGMENT (0.5-0.8): Add high-confidence edges to manual DAG
    - REVIEW (0.3-0.5): Flag for expert review
    - REJECT (< 0.3): Fall back to manual DAG
    """

    def evaluate(
        self,
        result: DiscoveryResult,
        expected_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> GateEvaluation:
        """
        Evaluate discovery result quality.

        Args:
            result: DiscoveryResult from runner
            expected_edges: Optional ground truth for validation

        Returns:
            GateEvaluation with decision, confidence, reasons
        """
        pass

    def should_accept(self, result: DiscoveryResult) -> bool:
        """Quick check if result should be accepted."""
        pass

    def get_augmentation_edges(
        self,
        result: DiscoveryResult,
        manual_dag: nx.DiGraph,
    ) -> List[DiscoveredEdge]:
        """Get high-confidence edges safe to add to manual DAG."""
        pass
```

#### DriverRanker
Compares causal vs predictive feature importance:
```python
class DriverRanker:
    """
    Compare causal importance (from DAG) vs predictive importance (from SHAP).

    Identifies:
    - Causal-only features: High causal rank, low predictive rank
    - Predictive-only features: High predictive rank, low causal rank
    - Concordant features: Similar ranks in both
    """

    def rank_drivers(
        self,
        dag: nx.DiGraph,
        target: str,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> DriverRankingResult:
        """
        Rank features by causal and predictive importance.

        Args:
            dag: Discovered or manual DAG
            target: Target variable for ranking
            shap_values: SHAP values matrix (n_samples, n_features)
            feature_names: Feature names corresponding to columns

        Returns:
            DriverRankingResult with rankings, correlation, categorization
        """
        pass

    def rank_from_discovery_result(
        self,
        result: DiscoveryResult,
        target: str,
        shap_values: np.ndarray,
    ) -> DriverRankingResult:
        """Convenience wrapper using DiscoveryResult."""
        pass
```

#### Discovery Data Types (base.py)
```python
class DiscoveryAlgorithmType(Enum):
    """Supported discovery algorithms."""
    GES = "ges"      # Greedy Equivalence Search
    PC = "pc"        # Peter-Clark
    FCI = "fci"      # Fast Causal Inference (future)
    LINGAM = "lingam"  # Linear Non-Gaussian (future)

class GateDecision(Enum):
    """Discovery gate decisions."""
    ACCEPT = "accept"
    AUGMENT = "augment"
    REVIEW = "review"
    REJECT = "reject"

class EdgeType(Enum):
    """Edge types in discovered graphs."""
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    BIDIRECTED = "bidirected"

@dataclass
class DiscoveredEdge:
    """Single discovered edge with metadata."""
    source: str
    target: str
    confidence: float = 1.0
    algorithm_votes: int = 1
    edge_type: EdgeType = EdgeType.DIRECTED

@dataclass
class DiscoveryConfig:
    """Configuration for discovery run."""
    algorithms: List[DiscoveryAlgorithmType] = field(default_factory=lambda: [
        DiscoveryAlgorithmType.GES,
        DiscoveryAlgorithmType.PC
    ])
    alpha: float = 0.05  # Significance level for CI tests
    ensemble_threshold: float = 0.5  # Min agreement for edge inclusion
    max_iterations: int = 1000
    cache_key: Optional[str] = None

@dataclass
class DiscoveryResult:
    """Result from discovery run."""
    success: bool
    config: DiscoveryConfig
    ensemble_dag: Optional[nx.DiGraph] = None
    edges: List[DiscoveredEdge] = field(default_factory=list)
    algorithm_results: List[AlgorithmResult] = field(default_factory=list)
    session_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GateEvaluation:
    """Result from gate evaluation."""
    decision: GateDecision
    confidence: float
    reasons: List[str]
    high_confidence_edges: List[DiscoveredEdge] = field(default_factory=list)
    rejected_edges: List[DiscoveredEdge] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Discovery Usage Example
```python
from src.causal_engine.discovery import (
    DiscoveryRunner,
    DiscoveryConfig,
    DiscoveryAlgorithmType,
    DiscoveryGate,
    DriverRanker,
)

# Run discovery
runner = DiscoveryRunner()
config = DiscoveryConfig(
    algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
    ensemble_threshold=0.5,
    alpha=0.05,
)
result = await runner.discover_dag(data, config)

# Evaluate with gate
gate = DiscoveryGate()
evaluation = gate.evaluate(result)
print(f"Decision: {evaluation.decision}")  # ACCEPT, REVIEW, REJECT, AUGMENT

# If AUGMENT, get safe edges to add
if evaluation.decision == GateDecision.AUGMENT:
    edges_to_add = gate.get_augmentation_edges(result, manual_dag)

# Rank drivers (if SHAP values available)
ranker = DriverRanker()
ranking = ranker.rank_from_discovery_result(result, target="outcome", shap_values=shap)
for r in ranking.get_by_causal_rank(top_k=5):
    print(f"{r.feature_name}: causal={r.causal_rank}, predictive={r.predictive_rank}")
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
